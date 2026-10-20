"""Stage 1: locality-aware, priority-guided UI exploration.

The explorer enumerates newly discovered states in breadth-first order.  To
enumerate one state it performs a bounded downward sweep until the visible
anchor set stops changing.  Getting to the state comes in two flavours: when
the device is already believed to be at it and no swipe has moved the viewport
since, the sweep starts straight from a verification capture; otherwise the app
is cold-started and the state's trusted navigation path is replayed first.  An
unverified warm screen is never swept — a mismatched verification capture falls
back to the cold-start path.  The resulting candidates include component
interactions, formal scroll actions, and Back on non-entry states.

Action execution is scheduled separately from state enumeration.  When the
device is already at a state with an unexplored action, the explorer prefers
that local action and avoids an unnecessary restart.  Otherwise it selects a
global candidate using replay depth, action-type weight, and action-type
attempt frequency, then cold-starts and replays only when needed.  Each trial
captures the live pre-state before executing the selected action and records
the observed post-state.

Drift is not a failure. If replay lands somewhere other than the
intended source state, we record the actual source plus the intended
one and move on. The FSM builder aggregates facts; the explorer does
not promise nav_path precision.

Sentinel ``target_state_id`` values flag non-transitional outcomes for
downstream filters: ``COLD_START_FAILED`` (couldn't foreground the
app), ``ACTION_FAILED`` (device API raised), ``LEFT_APP`` (ended up
outside the allowed app scope).  Refused actions are also persisted as
``ActionAttempt`` records and never become FSM transitions.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uiautomator2 as u2
from loguru import logger
from pydantic import BaseModel, Field

from vigil.core.config import VigilConfig
from vigil.core.ui_compressor import compact_ui_tree_text
from vigil.core.ui_parser import parse_hierarchy_xml
from vigil.core.ui_selectors import (
    build_component_selector,
    find_element_by_selector,
    selector_has_stable_identity,
    selector_identity,
)
from vigil.models.action import Action, ActionType
from vigil.models.state import (
    SENTINEL_ACTION_FAILED,
    SENTINEL_COLD_START_FAILED,
    SENTINEL_LEFT_APP,
    SENTINEL_TARGETS,
    RawScreen,
    UIElement,
    _normalize_dynamic,
    is_text_input,
)
from vigil.neuro.scope_policy import ScopeCategory, ScopePolicy
from vigil.neuro.selector_resolution import (
    ResolutionStatus,
    resolve_selector,
)

# Exploration priority weights. Click-equivalent actions default to 1.0;
# navigation / scroll actions are deprioritized because they tend to revisit
# already-known states rather than discover new ones.
ACTION_TYPE_WEIGHT: dict[ActionType, float] = {
    ActionType.CLICK: 1.0,
    ActionType.LONG_PRESS: 1.0,
    ActionType.INPUT_TEXT: 1.0,
    ActionType.NAVIGATE_BACK: 0.3,
    ActionType.NAVIGATE_HOME: 0.2,
    ActionType.SCROLL_DOWN: 0.5,
    ActionType.SCROLL_UP: 0.5,
}


# Fast foreground query. See ``AppExplorer._foreground_app`` for why this
# exists instead of ``uiautomator2``/``adbutils`` ``app_current()``.
#
# ``dumpsys window displays`` prints one ``mFocusedApp`` line per display, with
# the default display first; the focused app of display 0 is the one every
# foreground/scope decision in this class means, so a plain ``search`` (first
# match) is the intended semantics, not an accident.
#
#   mFocusedApp=ActivityRecord{ad0e348 u0 com.example.app/.MainActivity t92}
#                              ^hash    ^user ^package    ^activity  ^task
#
# The activity is captured in the package-relative **short** form
# (``.MainActivity``) — the same form ``app_current()`` reports. This matters:
# ``activity_name`` feeds ``RawScreen.get_hybrid_state_id``, so emitting a
# fully-qualified name here would change every recorded ``state_id``.
FOCUSED_APP_RE = re.compile(
    r"mFocusedApp=ActivityRecord\{\S+\s+\S+\s+(?P<package>[^\s/]+)/(?P<activity>[^\s]+)"
)


def action_key(action: Action) -> str:
    """Stable descriptor-based identity.

    When ``target_selector`` has stable identity (rid / text / content-desc /
    nearby-text), key = ``action_type|selector_identity``. Otherwise falls
    back to the descriptor tuple (rid, text, content-desc, class) so actions
    whose elements lack a stable selector still deduplicate consistently.
    Bounds are always excluded — the same logical element at
    different scroll offsets has different bounds in different captures.
    Use :func:`is_action_identifiable` to check whether an action is safe
    to enumerate (at least one selector or descriptor field non-empty).
    """
    if selector_has_stable_identity(action.target_selector):
        return f"{action.action_type.value}|{selector_identity(action.target_selector)}"
    return "|".join(
        [
            action.action_type.value,
            action.target_resource_id or "",
            action.target_text or "",
            action.target_content_desc or "",
            action.target_class_name or "",
        ]
    )


def is_action_identifiable(action: Action) -> bool:
    """True iff at least one descriptor or selector field is populated.

    Global actions (NAVIGATE_BACK, NAVIGATE_HOME) are always identifiable
    by type alone — they have no target and trivially deduplicate on
    action_type + empty descriptor.
    """
    if action.action_type in (ActionType.NAVIGATE_BACK, ActionType.NAVIGATE_HOME):
        return True
    if selector_has_stable_identity(action.target_selector):
        return True
    return bool(
        action.target_resource_id
        or action.target_text
        or action.target_content_desc
        or action.target_class_name
    )


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _short_target(action: Action) -> str:
    if action.target_resource_id:
        rid = action.target_resource_id
        return f"rid={rid.split(':id/', 1)[1] if ':id/' in rid else rid}"
    if action.target_text:
        t = action.target_text
        return f'text="{t[:30]}"'
    if action.target_content_desc:
        return f'cd="{action.target_content_desc[:30]}"'
    if action.target_class_name:
        return f"cls={action.target_class_name.rsplit('.', 1)[-1]}"
    if action.target_bounds:
        return f"bounds={action.target_bounds}"
    return "-"


def _descendant_title_texts(element: UIElement, by_id: dict[str, UIElement]) -> list[str]:
    """Normalized texts of ``android:id/title`` descendants, DFS order.

    Used to resolve Preference-row clickables whose label lives in a child
    TextView rather than on the clickable itself.
    """
    texts: list[str] = []
    stack = list(element.children)
    seen: set[str] = set()
    while stack:
        cid = stack.pop()
        if cid in seen:
            continue
        seen.add(cid)
        child = by_id.get(cid)
        if child is None:
            continue
        if child.resource_id == "android:id/title" and child.text:
            texts.append(_normalize_dynamic(child.text))
        stack.extend(child.children)
    return texts


def _descendant_title_text(element: UIElement, elements: list[UIElement]) -> str | None:
    """Return the normalized text of the first ``android:id/title`` descendant."""
    by_id = {e.element_id: e for e in elements}
    texts = _descendant_title_texts(element, by_id)
    return texts[0] if texts else None


def _is_edit_text(element: UIElement) -> bool:
    """Is this element an Android EditText / text-input widget?

    Delegates to :func:`vigil.models.state.is_text_input` so that enumeration
    here and selector resolution in ``selector_resolution._action_compatible``
    can never disagree about what is typeable.
    """
    return is_text_input(element)


#: Words that mark a field as filtering *existing* content rather than
#: authoring new content. Checked against the hint, the placeholder text and the
#: resource id.
_SEARCH_FIELD_HINTS = ("search", "filter", "query", "find")


def _looks_like_search_field(element: UIElement) -> bool:
    """Does this field filter existing content, rather than author new content?

    An empty Android text field reports its hint in ``text``, so
    ``'Search in ALL RECIPES'`` is a placeholder, not a value.
    """
    haystack = " ".join(
        (
            element.content_description or "",
            element.text or "",
            element.resource_id or "",
        )
    ).lower()
    return any(kw in haystack for kw in _SEARCH_FIELD_HINTS)


def _screen_derived_query(element: UIElement, elements: list[UIElement]) -> str | None:
    """A query drawn from content visible on this screen, or None.

    Typing a constant into a search box reaches a NO-RESULTS screen -- a new
    state, but a dead one. To reach the *results* screen the query has to match
    something that is actually there, so it is taken from the list rows on the
    current screen.

    A row field is identified structurally, not by name: a ``resource_id`` that
    occurs more than once on one screen is a repeated row slot (``card_text_view_title``
    on six recipe cards, and so on). The first such field in document order wins,
    which keeps the choice deterministic for a given screen.

    Returns the first whitespace-delimited token, so the query matches the whole
    row family rather than exactly one row; falls back to the full text when the
    leading token is too short to be selective.
    """
    counts: dict[str, int] = {}
    for e in elements:
        rid = e.resource_id
        if rid and (e.text or "").strip():
            counts[rid] = counts.get(rid, 0) + 1

    for e in elements:
        if e.element_id == element.element_id:
            continue
        rid, text = e.resource_id, (e.text or "").strip()
        if not rid or not text or counts.get(rid, 0) < 2:
            continue
        if _looks_like_search_field(e):
            continue
        token = text.split()[0] if text.split() else ""
        # Strip trailing punctuation so "Soup," does not become a dead query.
        token = token.strip(".,:;!?()[]{}\"'")
        if len(token) >= 3:
            return token
        return text
    return None


def _generate_edit_value(element: UIElement, elements: list[UIElement] | None = None) -> str:
    """Pick a plausible text value for a text-input widget.

    Priority:
      1. Search / filter fields take a token from repeated row content on the
         current screen, so the query matches and the *results* screen is
         reached rather than the empty-results screen.
      2. Hint / content-desc keyword matching (email, password, phone, ...).
      3. Generic ``test123`` fallback.

    ``elements`` is optional so existing callers keep working; without it rule 1
    cannot fire and the behaviour is exactly as before.

    Always returns a non-empty ASCII string.
    """
    if _looks_like_search_field(element):
        if elements:
            derived = _screen_derived_query(element, elements)
            if derived:
                return derived
        # No rows to draw from. Still a search field, so use the search constant
        # rather than falling through to ``test123``: the keyword table below
        # only inspects ``content_description``, and a search hint usually lives
        # in ``text``.
        return "test"

    hint = (element.content_description or "").lower()
    if any(kw in hint for kw in ("email", "e-mail")):
        return "test@example.com"
    if any(kw in hint for kw in ("password", "pwd", "pass")):
        return "TestPass123!"
    if any(kw in hint for kw in ("phone", "mobile", "tel")):
        return "5551234567"
    if any(kw in hint for kw in ("search", "query", "find")):
        return "test"
    if any(kw in hint for kw in ("url", "website", "link")):
        return "https://example.com"
    if "name" in hint:
        return "Test User"
    return "test123"


def _build_interact_action(element: UIElement, elements: list[UIElement]) -> Action:
    """Build an interaction action for an interactable element.

    Descriptor population:
      - ``target_resource_id`` from the element.
      - ``target_text`` from the element's own text, normalized. If the
        element has no rid/text/content_desc, borrow the first
        ``android:id/title`` descendant's text (Preference-row pattern).
      - ``target_content_desc`` from the element.
      - ``target_class_name`` from the element.

    Priority: INPUT_TEXT for EditText, LONG_PRESS for long-click-only
    elements, otherwise CLICK.

    See :func:`_build_interact_actions` for why a text field also yields a
    CLICK; this function returns the *primary* action only, and is kept as the
    single place descriptors are populated.
    """
    own_text = _normalize_dynamic(element.text) if element.text else ""
    cd = element.content_description or ""
    rid = element.resource_id or ""
    target_text = own_text
    if not rid and not own_text and not cd:
        borrowed = _descendant_title_text(element, elements)
        if borrowed:
            target_text = borrowed

    if _is_edit_text(element):
        return Action(
            action_type=ActionType.INPUT_TEXT,
            target_element_id=element.element_id,
            target_bounds=element.bounds,
            target_resource_id=element.resource_id,
            target_text=target_text or None,
            target_content_desc=cd or None,
            target_class_name=element.class_name or None,
            target_selector=build_component_selector(element, elements),
            input_text=_generate_edit_value(element, elements),
        )
    at = ActionType.CLICK
    if element.is_long_clickable and not element.is_clickable:
        at = ActionType.LONG_PRESS
    return Action(
        action_type=at,
        target_element_id=element.element_id,
        target_bounds=element.bounds,
        target_resource_id=element.resource_id,
        target_text=target_text or None,
        target_content_desc=cd or None,
        target_class_name=element.class_name or None,
        target_selector=build_component_selector(element, elements),
    )


def _build_interact_actions(element: UIElement, elements: list[UIElement]) -> list[Action]:
    """Every action this element affords, primary first.

    A text field is almost always *both* typeable and clickable -- measured over
    the AndroidWorld corpus, 3703 of 3727 text-input nodes (99.4%) carry
    ``clickable=true``. Returning only the primary action therefore drops a real
    affordance, and the two are not interchangeable: an agent clicks a search
    box to focus it and types into it as separate steps, so the graph needs both
    edges or one of those steps is unverifiable.

    This was already true for ordinary ``EditText`` fields; it only became
    visible when ``AutoCompleteTextView`` joined the text-input predicate and a
    measured target (a click on Broccoli's search box) lost its edge because the
    field started yielding ``INPUT_TEXT`` instead of ``CLICK``.

    Non-text elements are unaffected and still yield exactly one action, so
    enumeration for every previously-explored widget is unchanged.
    """
    primary = _build_interact_action(element, elements)
    if primary.action_type != ActionType.INPUT_TEXT or not element.is_clickable:
        return [primary]
    secondary = Action(
        action_type=ActionType.CLICK,
        target_element_id=primary.target_element_id,
        target_bounds=primary.target_bounds,
        target_resource_id=primary.target_resource_id,
        target_text=primary.target_text,
        target_content_desc=primary.target_content_desc,
        target_class_name=primary.target_class_name,
        target_selector=primary.target_selector,
    )
    return [primary, secondary]


def _build_scroll_down(screen: RawScreen) -> Action | None:
    """First scrollable element's descriptor drives the gesture. None if no scrollable."""
    for e in screen.elements:
        if e.is_scrollable and e.bounds and (e.bounds[3] - e.bounds[1]) > 0:
            return Action(
                action_type=ActionType.SCROLL_DOWN,
                target_element_id=e.element_id,
                target_bounds=e.bounds,
                target_resource_id=e.resource_id,
                target_class_name=e.class_name or None,
                target_selector=build_component_selector(e, screen.elements),
            )
    return None


class ExplorationTrace(BaseModel):
    """One observation record. Compatible with the previous trace schema;
    adds ``intended_source_state_id``, drops ``failure_reason``, and carries
    a free-form ``metadata`` dict for downstream diagnostics (selector
    resolution status, scope category, INPUT_TEXT bookkeeping). Older traces
    without metadata deserialize correctly because the field has a default
    factory.

    Invariant: a :class:`ExplorationTrace` represents an *executed*
    transition observation only. Actions that were refused before
    execution (ambiguous selector, out-of-scope element, INPUT_TEXT that could not
    guarantee clear-before-set) are recorded
    as :class:`ActionAttempt` records and MUST NOT appear in this list,
    so the FSM builder never turns them into transitions.
    """

    step_number: int
    source_state_id: str = ""
    intended_source_state_id: str = ""
    source_screen_id: str = ""
    action: Action
    target_state_id: str = ""
    target_screen_id: str = ""
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionAttempt(BaseModel):
    """A refused-before-execution attempt to interact with the device.

    Emitted (and persisted alongside :class:`ExplorationTrace`) when the
    explorer chooses not to execute an action, so post-hoc analysis can
    see *what* was refused and *why* without polluting the FSM with
    fake transitions. Never used to build FSM edges.
    """

    step_number: int
    source_state_id: str = ""
    source_screen_id: str = ""
    action: Action
    # Status values: "ambiguous_selector" | "skipped_unsafe_input"
    status: str
    reason: str = ""
    timestamp: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExplorationResult(BaseModel):
    app_package: str
    screens: dict[str, RawScreen] = Field(default_factory=dict)
    traces: list[ExplorationTrace] = Field(default_factory=list)
    total_steps: int = 0
    unique_screens: int = 0
    duration_seconds: float = 0.0
    output_dir: str = ""
    nav_stats: dict[str, int] = Field(default_factory=dict)


class AppExplorer:
    """Plain BFS explorer. See module docstring for rationale."""

    # Name stamped into the trace as ``exploration_strategy`` so a bundle always
    # declares how it was produced. A class attribute rather than a literal only
    # so a subclass can restamp it; the value written by this class is unchanged.
    STRATEGY_NAME: str = "bfs_restart_per_action"

    # Fixed post-action debounce. Two roles:
    #   1. the *floor* on every settle cap (see ``_wait_until_settled``), so a
    #      settle that runs out of budget can never leave a dump less settled
    #      than the blind sleep it replaced;
    #   2. the still-blind debounce after a swipe gesture (see
    #      ``POST_SWIPE_WAIT``), which the settle cannot replace.
    POST_ACTION_WAIT: float = 1.0
    # Debounce after a swipe. Deliberately NOT the adaptive settle: the settle
    # observable is the structural fingerprint (+ page title), and the
    # structural fingerprint drops the descendants of scrollable subtrees
    # (``models/state.py``). A fling therefore changes only what the observable
    # excludes, so two probes agree immediately and the settle returns at its
    # floor while the list is still moving — the next capture would enumerate a
    # half-scrolled list (under-collected actions) or resolve a target against
    # one. A blind wall-clock wait is the only thing here that actually covers
    # a fling, so both ``_swipe_scroll`` call sites keep it.
    POST_SWIPE_WAIT: float = POST_ACTION_WAIT
    WAIT_FOR_FOREGROUND: float = 10.0
    # Poll cadence for ``_wait_for_app_foreground``. Cold start no longer sleeps
    # a blind post-launch wait, so this poll is what actually detects the launch;
    # a tighter interval shortens every cold start by up to one interval without
    # changing the predicate or the cap.
    FOREGROUND_POLL_INTERVAL: float = 0.25
    # Upper bound (ms) we allow for the uiautomator server's *global*
    # wait-for-idle timeout. See ``_apply_wait_for_idle_cap``: the value is
    # clamped down to this, never raised, so a server that already waits less
    # keeps its faster setting.
    WAIT_FOR_IDLE_CAP_MS: int = 1500
    # Adaptive settle (replaces the blind POST_ACTION_WAIT sleep). Probe the
    # hierarchy after SETTLE_FLOOR, re-probe every SETTLE_INTERVAL, and give up
    # at the cap. SETTLE_CAP applies at transient sites (nav replay, scroll
    # waits); SETTLE_CAP_REGISTERED applies before a capture that gets
    # registered into the result's ``screens`` dict, where an unsettled dump
    # would silently mint a phantom state.
    # 0.25 s each: measured on the project emulator a settle that agrees on its
    # first two probes costs ~1.05 s end to end, i.e. what the 1.0 s blind sleep
    # cost. The safety of the settle comes from the mechanics (two agreeing
    # structural probes, an unstable reading on any failed probe, and a cap
    # floored at POST_ACTION_WAIT), not from the size of these two numbers.
    SETTLE_FLOOR: float = 0.25
    SETTLE_INTERVAL: float = 0.25
    SETTLE_CAP: float = 1.5
    SETTLE_CAP_REGISTERED: float = 4.0
    # Debounce after the home press only. It exists so the foreground poll
    # cannot false-positive on the app that was still resumed when the cold
    # start began; it is not a launch wait.
    COLD_START_HOME_WAIT: float = 0.2
    MAX_SCROLLS: int = 8
    MAX_SCROLL_TO_FIND: int = 10
    MAX_CONSECUTIVE_COLD_START_FAILURES: int = 5
    # How often one (state, action) may be re-offered after being picked but never
    # executed (the replay never reached the app). Small on purpose: each retry
    # costs one observation from the budget, and a state whose nav path is simply
    # unreplayable must not be able to consume the whole run. After this many
    # attempts the action is retired the old way so exploration keeps moving.
    MAX_UNEXECUTED_RETRIES: int = 2
    # Swipe geometry fallback when ``window_size()`` raises mid-scroll.
    FALLBACK_WINDOW_SIZE: tuple[int, int] = (1080, 2400)
    # Global animation scales zeroed for the duration of a run so dumps are
    # less likely to land mid-transition. Order is fixed for reproducible
    # metadata; the values are read back and restored in ``explore``'s
    # ``finally``.
    ANIMATION_SCALE_KEYS: tuple[str, ...] = (
        "window_animation_scale",
        "transition_animation_scale",
        "animator_duration_scale",
    )

    def __init__(
        self,
        device_serial: str,
        app_package: str,
        config: VigilConfig,
        output_dir: Path | None = None,
    ) -> None:
        self._serial = device_serial
        self._app_package = app_package
        self._config = config
        self._device: u2.Device | None = None
        self._screen_counter = 0
        if output_dir is None:
            self._output_dir = Path(f"data/apps/{app_package.replace('.', '_')}")
        else:
            self._output_dir = output_dir
        for sub in ("screens", "trees", "traces"):
            (self._output_dir / sub).mkdir(parents=True, exist_ok=True)

        # Action scheduling state is attached to the explorer so
        # _priority_score / _pick_next_action / _perform_one_observation
        # can read and mutate them without threading through arguments.
        # - _nav_paths: canonical replay nav_path from app entry per state_id
        # - _all_actions_per_state: enumerated outgoing actions, keyed by action_key
        # - _explored_per_state: action_keys already consumed (success or ACTION_FAILED)
        # - _blacklisted_per_state: action_keys skipped from future picks
        #   (LEFT_APP anchored on intended source; scroll-no-yield on actual source)
        # - _global_action_type_count: total execution attempts per ActionType
        self._nav_paths: dict[str, list[Action]] = {}
        self._all_actions_per_state: dict[str, dict[str, Action]] = {}
        self._explored_per_state: dict[str, set[str]] = defaultdict(set)
        self._blacklisted_per_state: dict[str, set[str]] = defaultdict(set)
        # (state_id, action_key) -> how often this action was picked but never
        # executed (the replay never reached the app). Bounds the re-queue.
        self._unexecuted_retries: dict[tuple[str, str], int] = defaultdict(int)
        self._global_action_type_count: dict[ActionType, int] = defaultdict(int)
        # Change 3 / 4: current-state tracker + emulator-health counter.
        # _current_state_id is None iff the device is in an unknown / non-app
        # state (initial, post-LEFT_APP, post-COLD_START_FAILED). When
        # non-None, the device is *believed* to be at that state_id — every
        # observation verifies this via pre-capture before executing.
        self._current_state_id: str | None = None
        self._consecutive_cold_start_failures: int = 0
        # Change 6 / viewport freshness: True when a swipe gesture may have left
        # the device at an unknown scroll offset for ``_current_state_id``.
        #
        # Tracked from ACTUAL gestures, not from the executed action's type:
        # every swipe in this class funnels through ``_swipe_scroll``, including
        # the scroll-to-find loop inside ``_resolve_action_target``, which fires
        # during ordinary CLICK observations. An action-type flag misses those
        # and would let a warm enumeration sweep a scrolled viewport.
        #
        # Cleared in exactly one place — a successful cold start — because that
        # is the only point where the viewport is *provably* fresh: the package
        # was force-stopped and relaunched, so every scrollable container is
        # back at its default offset.
        #
        # In particular it is NOT cleared when an action changes the observed
        # state. A changed hybrid state id does not imply a newly rendered
        # screen: an overlay / dialog / bottom sheet opening over a scrolled
        # list changes the id (it adds nodes outside the scrollable subtree)
        # while the list underneath stays where the last swipe left it. And
        # because the flag is already False whenever no swipe has happened, such
        # a clear can only ever fire after a real gesture — it has no sound
        # half. Leaving the flag set costs one cold start; clearing it wrongly
        # costs a silently under-enumerated state.
        # False at startup: the entry state is reached by the startup cold start.
        self._device_scrolled: bool = False

        # Generic single-app scope policy. No app-specific allowlists are introduced
        # here; defaults are Android framework packages.
        self._scope_policy: ScopePolicy = self._build_scope_policy()
        # Replay drift and verified navigation-path diagnostics.
        self._drift_count: int = 0
        self._verified_navigation_states: set[str] = set()
        self._unverified_navigation_targets: set[str] = set()
        # Side-channel for _execute_action diagnostics consumed by
        # _perform_one_observation. Reset at the start of every execution.
        self._last_execution_metadata: dict[str, Any] = {}
        # Refused-before-execution attempts. Saved alongside traces but
        # NEVER converted to FSM transitions.
        self._action_attempts: list[ActionAttempt] = []
        # Animation-scale override bookkeeping (see _disable_animation_scales).
        # ``None`` value = the key was unset on the device before the run and
        # must be restored with ``settings delete``, not ``settings put``.
        self._animation_scales_prior: dict[str, str | None] = {}
        self._animation_scales_overridden: bool = False
        # Adaptive-settle diagnostics. A cap hit means the screen never showed
        # two consecutive identical structural observations within the budget,
        # so the capture has the same observation limits as a fixed wait.
        # Emitted into ``nav_stats`` to expose how often settling reached its limit.
        #
        # `_settle_cap_hits_registered_cap` counts cap hits at the settle sites
        # that used SETTLE_CAP_REGISTERED, NOT captures that entered the screens
        # dict unsettled. It is a strict upper bound on the latter: those sites
        # are the ones that *may* be followed by a registered capture, but the
        # capture can also never happen (the post-action scope check bails to
        # LEFT_APP) or be transient (a cold start feeding an enumeration sweep).
        # Named for what it measures so the estimate is not read as an exact
        # count of unsettled evidence.
        self._settle_cap_hits: int = 0
        self._settle_cap_hits_registered_cap: int = 0

    @staticmethod
    def _build_scope_policy_from_defaults(app_package: str) -> ScopePolicy:
        from vigil.core.platform_priors import get_scope_defaults

        cfg = get_scope_defaults()
        kwargs: dict[str, Any] = {"app_package": app_package}
        if cfg.get("android_system_packages"):
            kwargs["android_system_packages"] = frozenset(cfg["android_system_packages"])
        if cfg.get("system_ui_packages"):
            kwargs["system_ui_packages"] = frozenset(cfg["system_ui_packages"])
        if cfg.get("launcher_patterns"):
            kwargs["launcher_patterns"] = tuple(cfg["launcher_patterns"])
        return ScopePolicy(**kwargs)

    def _build_scope_policy(self) -> ScopePolicy:
        return self._build_scope_policy_from_defaults(self._app_package)

    # ------------------------------------------------------------------ public

    def explore(self) -> ExplorationResult:
        budget = self._config.app.max_exploration_steps
        logger.info(
            f"Budget: {budget} observations — est. {max(1, round(budget * 25 / 60))} min wall-clock"
        )
        started = time.monotonic()
        try:
            self._connect_device()
            if not self._cold_start_app():
                raise RuntimeError(f"Could not launch {self._app_package}")

            entry = self._capture_screen()
            if entry is None:
                raise RuntimeError("Initial capture failed after cold-start")
            entry_sid = entry.get_hybrid_state_id(self._app_package)
            self._current_state_id = entry_sid
            self._nav_paths[entry_sid] = []
            self._verified_navigation_states.add(entry_sid)
            screens: dict[str, RawScreen] = {entry.screen_id: entry}
            traces: list[ExplorationTrace] = []
            counts = {"cold_start_failures": 0, "left_app": 0, "action_failures": 0}
            enumerate_queue: deque[str] = deque([entry_sid])
            done = 0

            while done < budget:
                # Enumerate one pending state per outer iteration so priority
                # selection always has up-to-date candidates. We enumerate the
                # shallowest unenumerated state (BFS-ordered discovery).
                unenumerated = [
                    sid for sid in enumerate_queue if sid not in self._all_actions_per_state
                ]
                if unenumerated:
                    unenumerated.sort(key=lambda s: len(self._nav_paths[s]))
                    sid = unenumerated[0]
                    enumerate_queue.remove(sid)
                    # _enumerate_actions_for_state owns the tracker across the
                    # sweep: it invalidates it whenever the sweep swiped (the
                    # viewport is left at an unknown offset whose state_id is
                    # scroll-blind), and restores it only in the zero-swipe
                    # case where the device provably never moved.
                    actions = self._enumerate_actions_for_state(sid, self._nav_paths[sid])
                    self._all_actions_per_state[sid] = {action_key(a): a for a in actions}
                    logger.info(f"state={sid[:6]} enumerated {len(actions)} actions")

                picked = self._pick_next_action()
                if picked is None:
                    if enumerate_queue:
                        continue  # more states awaiting enumeration
                    logger.info("Exploration complete: no more unexplored actions")
                    break

                state_id, action = picked
                done += 1
                t0 = time.monotonic()
                trace, pre_screen, post_screen = self._perform_one_observation(
                    intended_source_state_id=state_id,
                    intended_nav_path=self._nav_paths[state_id],
                    action=action,
                    step=done,
                )
                elapsed = time.monotonic() - t0
                self._log_step(trace, len(self._nav_paths[state_id]), elapsed)
                traces.append(trace)

                key = action_key(action)
                requeued = self._retire_or_requeue(state_id, key, trace)
                if requeued:
                    counts["unexecuted_requeued"] = counts.get("unexecuted_requeued", 0) + 1
                # The warm-enumeration gate (`_device_scrolled`) is maintained
                # at the gesture choke point (`_swipe_scroll`) and cleared only
                # on a successful cold start (`_cold_start_attempt`), not from
                # the action type here — scroll-to-find swipes during CLICK
                # observations are invisible to the action type.

                if trace.target_state_id == SENTINEL_COLD_START_FAILED:
                    counts["cold_start_failures"] += 1
                    self._consecutive_cold_start_failures += 1
                    if (
                        self._consecutive_cold_start_failures
                        >= self.MAX_CONSECUTIVE_COLD_START_FAILURES
                    ):
                        logger.error(
                            f"Aborting exploration: "
                            f"{self._consecutive_cold_start_failures} consecutive "
                            f"COLD_START_FAILED (last at source {state_id[:6]}). "
                            f"Device is persistently unresponsive. Unused budget: "
                            f"{budget - done} observations."
                        )
                        counts["aborted_on_cold_start_failures"] = 1
                        break
                elif trace.target_state_id == SENTINEL_LEFT_APP:
                    counts["left_app"] += 1
                    # Blacklist only what was actually retired. A genuine pre-capture
                    # means the action executed and the app then exited, which IS
                    # evidence about this state; a re-queued row proved only that the
                    # replay never arrived, and blacklisting it would retire a
                    # candidate that was never tried.
                    if not requeued:
                        self._blacklisted_per_state[state_id].add(key)
                    self._consecutive_cold_start_failures = 0
                elif trace.target_state_id == SENTINEL_ACTION_FAILED:
                    counts["action_failures"] += 1
                    self._consecutive_cold_start_failures = 0
                else:
                    # Any successful observation resets the consecutive counter.
                    # Keep this explicit so the invariant is obvious on inspection.
                    self._consecutive_cold_start_failures = 0
                    # Scroll-no-yield check: identical anchors pre/post → blacklist
                    # at ACTUAL source so we don't try the dead scroll again if the
                    # replay lands on a truly stable page.
                    if (
                        action.action_type in (ActionType.SCROLL_UP, ActionType.SCROLL_DOWN)
                        and pre_screen is not None
                        and post_screen is not None
                    ):
                        _, pre_anchors = pre_screen.get_functional_state_key(self._app_package)
                        _, post_anchors = post_screen.get_functional_state_key(self._app_package)
                        if pre_anchors == post_anchors:
                            self._blacklisted_per_state[trace.source_state_id].add(key)

                if pre_screen is not None:
                    screens[pre_screen.screen_id] = pre_screen
                if post_screen is not None:
                    screens[post_screen.screen_id] = post_screen

                tgt = trace.target_state_id
                if tgt and tgt not in SENTINEL_TARGETS:
                    actual_src = trace.source_state_id
                    drifted = bool(actual_src) and actual_src != state_id
                    framework_target = bool(trace.metadata.get("framework_scope"))
                    framework_source = (
                        trace.metadata.get("scope_pre") == ScopeCategory.ANDROID_SYSTEM.value
                    )
                    if drifted:
                        self._drift_count += 1
                    # Record successful IN_APP observations as verified navigation sources.
                    # ANDROID_SYSTEM screens are observation-only by default —
                    # neither the source nor the target is promoted to a
                    # verified state; they are not used as a basis for new
                    # nav paths because their content is transient (dialogs,
                    # permission prompts, document pickers).
                    if actual_src and not drifted and not framework_source:
                        self._verified_navigation_states.add(actual_src)
                    if tgt not in self._nav_paths:
                        if framework_target:
                            # Allow observation/capture (already done above
                            # via screens), but do NOT enqueue the
                            # ANDROID_SYSTEM target as a normal app state.
                            self._unverified_navigation_targets.add(tgt)
                        elif not drifted and not framework_source:
                            # Trusted: extend the intended source's nav_path.
                            self._nav_paths[tgt] = [*self._nav_paths[state_id], action]
                            enumerate_queue.append(tgt)
                            self._verified_navigation_states.add(tgt)
                        elif (
                            not framework_source
                            and actual_src in self._nav_paths
                            and actual_src in self._verified_navigation_states
                        ):
                            # Drift, but actual_src is verified IN_APP — extend
                            # that path. ANDROID_SYSTEM actual sources never
                            # provide a verified navigation basis.
                            self._nav_paths[tgt] = [*self._nav_paths[actual_src], action]
                            enumerate_queue.append(tgt)
                            self._verified_navigation_states.add(tgt)
                        else:
                            # Observed but not enqueued — explorer cannot
                            # reach `tgt` via a verified replay path.
                            self._unverified_navigation_targets.add(tgt)

            self._log_summary()
            return self._finalize(
                traces, screens, self._nav_paths, counts, done, time.monotonic() - started
            )
        finally:
            self._restore_animation_scales()

    # ---------------------------------------------------------- scheduling

    def _priority_score(self, state_id: str, action: Action) -> float:
        """Higher is better. Depth × type weight × log-frequency decay."""
        nav_depth = len(self._nav_paths.get(state_id, []))
        type_weight = ACTION_TYPE_WEIGHT.get(action.action_type, 1.0)
        global_freq = self._global_action_type_count[action.action_type]
        depth_factor = 1.0 / (1.0 + nav_depth)
        freq_factor = 1.0 / (1.0 + math.log1p(global_freq))
        return depth_factor * type_weight * freq_factor

    def _pick_next_action(self) -> tuple[str, Action] | None:
        """Locality preference + global argmax fallback.

        If ``self._current_state_id`` has an unexplored non-blacklisted
        action, pick the argmax *within* that state — skips a cold-start
        + nav-path replay next turn. Otherwise fall through to the global
        argmax across all enumerated states.
        """
        sid = self._current_state_id
        if sid is not None and sid in self._all_actions_per_state:
            local = self._pick_best_in_state(sid)
            if local is not None:
                return (sid, local)
        return self._global_argmax()

    def _pick_best_in_state(self, state_id: str) -> Action | None:
        """Argmax of _priority_score within a single state's unexplored,
        non-blacklisted outgoing actions. Returns None if empty."""
        actions = self._all_actions_per_state.get(state_id, {})
        if not actions:
            return None
        explored = self._explored_per_state[state_id]
        blacklisted = self._blacklisted_per_state[state_id]
        best: Action | None = None
        best_score = 0.0
        for key, action in actions.items():
            if key in explored or key in blacklisted:
                continue
            score = self._priority_score(state_id, action)
            if score > best_score:
                best_score = score
                best = action
        return best

    def _global_argmax(self) -> tuple[str, Action] | None:
        """Argmax across all enumerated states' unexplored non-blacklisted
        outgoing actions."""
        best: tuple[str, Action] | None = None
        best_score = 0.0
        for state_id, actions in self._all_actions_per_state.items():
            explored = self._explored_per_state[state_id]
            blacklisted = self._blacklisted_per_state[state_id]
            for key, action in actions.items():
                if key in explored or key in blacklisted:
                    continue
                score = self._priority_score(state_id, action)
                if score > best_score:
                    best_score = score
                    best = (state_id, action)
        return best

    def _retire_or_requeue(self, state_id: str, key: str, trace: ExplorationTrace) -> bool:
        """Consume the attempted action, or put it back on the frontier.

        Returns True when the action was re-queued (left unexplored and
        unblacklisted, so ``_pick_next_action`` will offer it again).

        An observation whose action never executed says nothing about this state.
        ``_perform_one_observation`` emits ``LEFT_APP`` with no ``source_screen_id``
        when the pre-capture found the device outside the app: the cold start + nav
        replay never arrived, so the row records *our* navigation failure, not the
        app's behaviour. Retiring the action on that basis consumes a candidate
        that was never tried, and a state whose whole frontier is consumed that way
        ends up with an empty outgoing set that reads, wrongly, as "every action
        here is forbidden".

        The retry is bounded. Each attempt costs one observation from the budget,
        so a state whose nav path is simply unreplayable must not be able to
        consume the run; after ``MAX_UNEXECUTED_RETRIES`` the action is retired the
        old way and exploration moves on. The next attempt does cold-start (the
        failure path already cleared ``_current_state_id``), so a re-queue has a
        real chance of landing.
        """
        never_executed = trace.target_state_id == SENTINEL_LEFT_APP and not trace.source_screen_id
        if not never_executed:
            self._explored_per_state[state_id].add(key)
            return False
        self._unexecuted_retries[(state_id, key)] += 1
        attempts = self._unexecuted_retries[(state_id, key)]
        if attempts > self.MAX_UNEXECUTED_RETRIES:
            logger.info(
                f"retire {key[:48]} at {state_id[:6]}: never executed in "
                f"{attempts} attempts, giving up"
            )
            self._explored_per_state[state_id].add(key)
            return False
        logger.info(
            f"re-queue {key[:48]} at {state_id[:6]}: action never executed "
            f"(attempt {attempts}/{self.MAX_UNEXECUTED_RETRIES})"
        )
        return True

    def _log_summary(self) -> None:
        """Summarize action attempts and the most frequently blacklisted states."""
        logger.info("Summary: action-type execution counts")
        for at, n in sorted(self._global_action_type_count.items(), key=lambda kv: -kv[1]):
            logger.info(f"  {at.value:<14} {n}")
        top = sorted(self._blacklisted_per_state.items(), key=lambda kv: -len(kv[1]))[:5]
        if top:
            logger.info("Summary: top blacklisted states")
            for sid, keys in top:
                if keys:
                    logger.info(f"  {sid[:6]}: {len(keys)} blacklisted")

    # ---------------------------------------------------------- core phases

    def _enumerate_actions_for_state(self, state_id: str, nav_path: list[Action]) -> list[Action]:
        """Restart + navigate + scroll-sweep to collect all candidate actions.

        Emits the formal Vigil action space for the current screen:
          - CLICK / LONG_PRESS / INPUT_TEXT for interactable elements
            (built by :func:`_build_interact_action`).
          - SCROLL_DOWN / SCROLL_UP per scrollable container as *formal*
            candidate actions (so the FSM records them as transitions /
            affordances even when source == target).
          - NAVIGATE_BACK as a global navigation action for non-entry
            states reached through a replay nav path.

        The internal scroll *sweep* used to reveal offscreen widgets is
        distinct from the formal SCROLL *candidates* emitted into the
        returned action list.

        SYSTEM_UI elements are filtered out — they belong to status / nav
        bar overlays and must not contribute to app-level identity or
        action enumeration.

        Warm enumeration (Change 6): when the device is already believed to be
        at ``state_id`` and no swipe gesture has moved the viewport since it was
        last known fresh (``_device_scrolled``), the cold start + nav replay is
        skipped and the sweep starts from a verification capture. Two
        independent checks gate this, because neither alone is sufficient:

        * ``_device_scrolled`` covers the scroll offset. It is the only one that
          can: it is maintained at the gesture choke point ``_swipe_scroll``, so
          it sees the scroll-to-find swipes that happen inside otherwise
          ordinary CLICK observations.
        * the verification capture's ``get_hybrid_state_id`` must equal
          ``state_id`` — this catches arriving at the wrong screen entirely, but
          NOT a wrong scroll offset (the hybrid id is identical at the top and
          the bottom of a scrolled page by construction).

        On either failure the method falls back to the full cold-start path — an
        unverified warm screen is never swept. The verification capture is reused
        as sweep iteration 1, so the warm path costs no extra dump, and (like
        every other enumeration capture) it is never registered into the result's
        ``screens`` dict.

        On return the current-state tracker is restored to ``state_id`` only
        when the sweep issued zero swipe gestures over a screen it actually saw
        and nothing else swiped in the meantime; in every other case it is
        invalidated. A swept sweep leaves the viewport at an offset that the
        hybrid state id cannot distinguish from the top of the page, and trusting
        it would send the next pick at an element that has scrolled out of reach
        (a permanent ACTION_FAILED, i.e. a silently lost edge).
        """
        warm_screen: RawScreen | None = None
        if self._current_state_id == state_id and not self._device_scrolled:
            candidate = self._capture_screen(take_screenshot=False)
            if candidate is not None and (
                candidate.get_hybrid_state_id(self._app_package) == state_id
            ):
                warm_screen = candidate
            else:
                logger.info(
                    f"warm enumerate of {state_id[:6]} unverified "
                    f"({'capture failed' if candidate is None else 'state mismatch'}) "
                    "— falling back to cold-start"
                )

        if warm_screen is None:
            self._current_state_id = None
            if not self._cold_start_app():
                logger.warning(f"cold-start failed during enumerate of {state_id[:6]}")
                return []
            for step in nav_path:
                # Replay: no `screen=` — every step must resolve against a fresh
                # capture because the device state changes at each step. The
                # settle is what makes that fresh capture usable under a capped
                # server-side wait-for-idle: the dump no longer blocks until the
                # screen is quiet, so we establish quiet ourselves before
                # resolving. A SCROLL step additionally gets the blind
                # post-swipe debounce the settle cannot provide.
                self._execute_action(step)
                self._wait_after_action(step, cap=self.SETTLE_CAP)

        collected: dict[str, Action] = {}
        saw_source_screen = False
        last_anchors: frozenset[tuple[str, str]] | None = None
        swipes = 0
        pending: RawScreen | None = warm_screen

        screen: RawScreen | None
        for _ in range(self.MAX_SCROLLS):
            if pending is not None:
                screen, pending = pending, None
            else:
                screen = self._capture_screen(take_screenshot=False)
            if screen is None:
                break
            saw_source_screen = True
            for e in screen.elements:
                # SYSTEM_UI elements never become app actions.
                if self._scope_policy.should_filter_element(e.package):
                    continue
                interactable = (
                    e.is_clickable or e.is_long_clickable or e.is_checkable or e.is_editable
                )
                scrollable_candidate = e.is_scrollable and bool(e.bounds)
                if not (interactable or scrollable_candidate):
                    continue
                pkg = (e.package or "").strip()
                if pkg:
                    cat = self._scope_policy.classify(pkg)
                    # Allow app and Android framework elements during exploration
                    # to be enumerated, since framework dialogs / pickers
                    # are part of legitimate in-app flows. SYSTEM_UI was
                    # already filtered above.
                    if cat not in (ScopeCategory.IN_APP, ScopeCategory.ANDROID_SYSTEM):
                        continue
                if interactable:
                    for action in _build_interact_actions(e, screen.elements):
                        if not is_action_identifiable(action):
                            continue
                        collected.setdefault(action_key(action), action)
                if scrollable_candidate:
                    # Emit formal SCROLL_DOWN / SCROLL_UP candidates per
                    # scrollable container. These persist in the FSM as
                    # transitions, including the source==target case which
                    # is meaningful for offscreen coverage.
                    for st in (ActionType.SCROLL_DOWN, ActionType.SCROLL_UP):
                        scroll_action = Action(
                            action_type=st,
                            target_element_id=e.element_id,
                            target_bounds=list(e.bounds),
                            target_resource_id=e.resource_id,
                            target_class_name=e.class_name or None,
                            target_selector=build_component_selector(e, screen.elements),
                        )
                        if is_action_identifiable(scroll_action):
                            collected.setdefault(action_key(scroll_action), scroll_action)

            _, anchors = screen.get_functional_state_key(self._app_package)
            if last_anchors is not None and anchors == last_anchors:
                break
            last_anchors = anchors
            scroll = _build_scroll_down(screen)
            if scroll is None:
                break
            # Counted before the gesture and regardless of its outcome: a swipe
            # that partially applied still moves the viewport.
            swipes += 1
            self._swipe_scroll(ActionType.SCROLL_DOWN, scroll)
            # Blind wall-clock debounce, NOT the settle: the settle cannot see a
            # fling (see POST_SWIPE_WAIT). Enumerating the next iteration off a
            # still-moving list under-collects actions — a silent coverage loss.
            time.sleep(self.POST_SWIPE_WAIT)

        # Tracker restoration is the zero-swipe case only (see docstring). A
        # sweep that never saw a screen tells us nothing about where the device
        # is, so it invalidates too; so does any gesture that fired outside the
        # sweep's own counter (e.g. scroll-to-find during the nav replay above).
        restorable = swipes == 0 and saw_source_screen and not self._device_scrolled
        self._current_state_id = state_id if restorable else None

        if nav_path and saw_source_screen:
            back_action = Action(action_type=ActionType.NAVIGATE_BACK)
            collected.setdefault(action_key(back_action), back_action)
        return list(collected.values())

    def _perform_one_observation(
        self,
        intended_source_state_id: str,
        intended_nav_path: list[Action],
        action: Action,
        step: int,
    ) -> tuple[ExplorationTrace, RawScreen | None, RawScreen | None]:
        """One observation. Skips cold-start + replay when the device is
        already at the intended source state; otherwise performs the full
        cycle. See module docstring for sentinels and the ``_current_state_id``
        invariant.
        """
        ts = _now_iso()

        def _trace(
            *,
            src: str,
            src_screen_id: str = "",
            target: str,
            target_screen_id: str = "",
            extra_metadata: dict[str, Any] | None = None,
        ) -> ExplorationTrace:
            md: dict[str, Any] = dict(extra_metadata or {})
            return ExplorationTrace(
                step_number=step,
                intended_source_state_id=intended_source_state_id,
                source_state_id=src,
                source_screen_id=src_screen_id,
                action=action,
                target_state_id=target,
                target_screen_id=target_screen_id,
                timestamp=ts,
                metadata=md,
            )

        needs_nav = self._current_state_id != intended_source_state_id
        if needs_nav:
            if not self._cold_start_app():
                self._current_state_id = None
                return (
                    _trace(src=intended_source_state_id, target=SENTINEL_COLD_START_FAILED),
                    None,
                    None,
                )
            for nav_step in intended_nav_path:
                # Replay: no `screen=` — each step lands on a new screen. The
                # settle stands in for the fixed debounce so the next step (and
                # ultimately the registered `pre` capture) resolves against a
                # screen that has stopped moving. On a cap hit it waits longer
                # than the old sleep did, never less.
                self._execute_action(nav_step)
                self._wait_after_action(nav_step, cap=self.SETTLE_CAP)

        pre = self._capture_screen()
        if pre is None:
            # No evidence screen exists for this row, so there is nothing to
            # derive the scope from: keep querying the device live. Never
            # record classify(None) (== OUT_OF_SCOPE_EXTERNAL) as scope_pre
            # here — that would assert an external package we never observed.
            pre_scope = self._current_scope()
            self._current_state_id = None
            ext_pkg = self._current_foreground_package()
            ext_md: dict[str, Any] = {
                "scope_pre": pre_scope.value,
            }
            if pre_scope == ScopeCategory.OUT_OF_SCOPE_EXTERNAL:
                ext_md["external_package"] = ext_pkg
                ext_md["left_app_reason"] = "out_of_scope_external"
            elif pre_scope == ScopeCategory.LAUNCHER_OR_HOME:
                ext_md["left_app_reason"] = "launcher_or_home"
                ext_md["external_package"] = ext_pkg
            return (
                _trace(
                    src=intended_source_state_id,
                    target=SENTINEL_LEFT_APP,
                    extra_metadata=ext_md,
                ),
                None,
                None,
            )
        # scope_pre is derived from the evidence screen itself rather than a
        # second app_current() round-trip: same classification the live query
        # produced, minus the window in which the foreground could change
        # between the dump and the re-query. `_scope_of_screen` (not a bare
        # classify) because a capture may legitimately carry an *empty*
        # package, which classify() would call OUT_OF_SCOPE_EXTERNAL.
        pre_scope = self._scope_of_screen(pre)
        actual_src = pre.get_hybrid_state_id(self._app_package)

        # Locality drift fallback: we thought we were at intended but the
        # live capture says otherwise (e.g., a system dialog surfaced).
        # Re-run this observation via the full cold-start path for
        # correctness. Log as INFO, not failure.
        if not needs_nav and actual_src != intended_source_state_id:
            cur_tag = self._current_state_id[:6] if self._current_state_id else "-"
            logger.info(
                f"locality drift: current={cur_tag} actual={actual_src[:6]} "
                f"intended={intended_source_state_id[:6]} — falling back to cold-start"
            )
            self._current_state_id = None
            if not self._cold_start_app():
                return (
                    _trace(src=intended_source_state_id, target=SENTINEL_COLD_START_FAILED),
                    None,
                    None,
                )
            for nav_step in intended_nav_path:
                # Replay: no `screen=` — each step lands on a new screen. The
                # settle stands in for the fixed debounce so the next step (and
                # ultimately the registered `pre` capture) resolves against a
                # screen that has stopped moving. On a cap hit it waits longer
                # than the old sleep did, never less.
                self._execute_action(nav_step)
                self._wait_after_action(nav_step, cap=self.SETTLE_CAP)
            pre = self._capture_screen()
            if pre is None:
                return (
                    _trace(src=intended_source_state_id, target=SENTINEL_LEFT_APP),
                    None,
                    None,
                )
            actual_src = pre.get_hybrid_state_id(self._app_package)
            pre_scope = self._scope_of_screen(pre)

        # Count attempts per action_type rather than successes, so the
        # frequency decay can't be gamed by a misbehaving type that always
        # fails.
        self._global_action_type_count[action.action_type] += 1

        # Resolve the action against `pre` — the very screen this row records
        # as source_screen_id. No device input happens between the pre dump and
        # this call, so the reused screen can never be staler than the recorded
        # evidence, and target_element_id linkage now points into the same
        # capture that fsm_builder.build_trace_action_sample and validate_fsm
        # resolve against. On the drift path `pre` is the post-cold-start
        # recapture (rebound above), never the pre-drift capture.
        #
        # Honest note on the foreground race: previously a foreground loss
        # between the pre dump and execution made the resolve-time capture
        # return None, yielding ACTION_FAILED with no click. Under reuse the
        # click fires blind and the loss is caught post-action as LEFT_APP.
        # These are not byte-identical; both are structurally filtered
        # sentinels, so no edge is fabricated either way.
        executed = self._execute_action(action, screen=pre)
        exec_md = dict(self._last_execution_metadata)
        if not executed:
            self._current_state_id = None
            # Distinguish *refused* attempts (ambiguous selector, unsafe
            # INPUT_TEXT) from *failed* executions. Refusals are not FSM
            # observations; record them as ActionAttempts and emit a
            # SENTINEL trace only as a placeholder that FsmBuilder filters.
            refusal_reason = ""
            input_policy = exec_md.get("input_policy")
            if exec_md.get("selector_resolution") == "ambiguous":
                refusal_reason = "ambiguous_selector"
            elif input_policy == "skipped_unsafe":
                refusal_reason = "skipped_unsafe_input"
            if refusal_reason:
                self._action_attempts.append(
                    ActionAttempt(
                        step_number=step,
                        source_state_id=actual_src,
                        source_screen_id=pre.screen_id,
                        action=action,
                        status=refusal_reason,
                        reason=exec_md.get("selector_resolution_reason", "")
                        if refusal_reason == "ambiguous_selector"
                        else "INPUT_TEXT could not guarantee clear-before-set",
                        timestamp=_now_iso(),
                        metadata=exec_md,
                    )
                )
            return (
                _trace(
                    src=actual_src,
                    src_screen_id=pre.screen_id,
                    target=SENTINEL_ACTION_FAILED,
                    extra_metadata={"scope_pre": pre_scope.value, **exec_md},
                ),
                pre,
                None,
            )
        # Registered-capture settle: `post` below goes into the result's
        # `screens` dict, so an unsettled dump here becomes a phantom state with
        # no detector downstream. The larger cap means a give-up still leaves
        # more settling than the fixed 1.0 s sleep this replaced. The scope
        # check deliberately sits *after* the settle: a probe that lands on a
        # transient launcher / system frame keeps the loop running, so scope is
        # only classified once the screen holds still.
        #
        # `_wait_after_action` (not a bare settle): when `action` was a SCROLL,
        # the swipe just fired and the settle is fling-blind, so `post` — a
        # registered evidence screen, and the input to the scroll-no-yield
        # blacklist check in `explore` — would otherwise be dumped mid-fling.
        self._wait_after_action(action, cap=self.SETTLE_CAP_REGISTERED, registered_cap=True)

        post_scope = self._current_scope()
        if not self._scope_policy.is_allowed(post_scope):
            self._current_state_id = None
            md: dict[str, Any] = {
                "scope_pre": pre_scope.value,
                "scope_post": post_scope.value,
                **exec_md,
            }
            if post_scope == ScopeCategory.OUT_OF_SCOPE_EXTERNAL:
                md["external_package"] = self._current_foreground_package()
                md["left_app_reason"] = "out_of_scope_external"
            elif post_scope == ScopeCategory.LAUNCHER_OR_HOME:
                md["external_package"] = self._current_foreground_package()
                md["left_app_reason"] = "launcher_or_home"
            elif post_scope == ScopeCategory.SYSTEM_UI:
                md["left_app_reason"] = "system_ui"
            return (
                _trace(
                    src=actual_src,
                    src_screen_id=pre.screen_id,
                    target=SENTINEL_LEFT_APP,
                    extra_metadata=md,
                ),
                pre,
                None,
            )
        post = self._capture_screen()
        if post is None:
            self._current_state_id = None
            return (
                _trace(
                    src=actual_src,
                    src_screen_id=pre.screen_id,
                    target=SENTINEL_LEFT_APP,
                    extra_metadata={
                        "scope_pre": pre_scope.value,
                        "scope_post": post_scope.value,
                        **exec_md,
                    },
                ),
                pre,
                None,
            )
        target_sid = post.get_hybrid_state_id(self._app_package)
        # Viewport-freshness bookkeeping for the warm-enumeration gate: nothing
        # is cleared here, deliberately. A post-action clear gated on "the state
        # id changed" was removed because a changed hybrid state id does NOT
        # imply a newly rendered screen — an overlay, dialog, or bottom sheet
        # opening on top of a scrolled list adds nodes outside the scrollable
        # subtree, so the id changes while the list underneath stays exactly
        # where the last swipe left it. Both swept apps reach this directly
        # (chat thread -> message_options, market cart -> remove_dialog).
        #
        # It also had no sound half: when no swipe has occurred the flag is
        # already False and the assignment is a no-op, so the *only* time it had
        # any effect was after a gesture had genuinely moved the viewport, i.e.
        # it could only ever clear a correctly-set flag. A successful cold start
        # (``_cold_start_attempt``) is the one unconditionally sound reset, and
        # is now the only one.
        self._current_state_id = target_sid
        return (
            _trace(
                src=actual_src,
                src_screen_id=pre.screen_id,
                target=self._current_state_id,
                target_screen_id=post.screen_id,
                extra_metadata={
                    "scope_pre": pre_scope.value,
                    "scope_post": post_scope.value,
                    "framework_scope": self._scope_policy.is_framework_scope(post_scope),
                    **exec_md,
                },
            ),
            pre,
            post,
        )

    def _log_step(self, trace: ExplorationTrace, depth: int, elapsed: float) -> None:
        drift = (
            "(as expected)"
            if trace.source_state_id == trace.intended_source_state_id
            else f"(drifted from {trace.intended_source_state_id[:6]})"
        )
        tgt = trace.target_state_id
        tgt_short = tgt if tgt in SENTINEL_TARGETS else tgt[:6]
        logger.info(
            f"step {trace.step_number} | replay({depth}) -> "
            f"src={trace.source_state_id[:6]} {drift} | "
            f"{trace.action.action_type.value} {_short_target(trace.action)} | "
            f"tgt={tgt_short} | {elapsed:.1f}s"
        )

    # ---------------------------------------------------------- device I/O

    def _connect_device(self) -> None:
        logger.info(f"Connecting to device: {self._serial}")
        self._device = u2.connect(self._serial)
        info = self._device.info
        logger.info(
            f"Connected: {info.get('productName', 'unknown')} (SDK {info.get('sdkInt', '?')})"
        )
        self._disable_animation_scales()
        self._apply_wait_for_idle_cap()

    def _shell_output(self, command: str) -> str:
        """Run ``command`` on the device and return its stdout as stripped text.

        ``uiautomator2`` returns a ``ShellResponse(output, exit_code)`` named
        tuple; looser shells return a bare string. Both are accepted. Device
        errors propagate — every caller here is already inside a ``try``.
        """
        assert self._device is not None
        raw = self._device.shell(command)
        return str(getattr(raw, "output", raw) or "").strip()

    def _foreground_app(self) -> tuple[str, str]:
        """Return ``(package, activity)`` of the focused app. Fast path first.

        **This is a workaround for an adbutils / Android-14 incompatibility, not
        a gratuitous reimplementation.** ``uiautomator2``'s ``app_current()``
        delegates to ``adbutils.shell.app_current``, whose three-step cascade
        was traced end to end on this project's Pixel_6a / API 34 emulator:

        1. ``dumpsys window windows`` (0.07 s, 60 KB) — this system image emits
           **no ``mCurrentFocus`` line at all**, so the primary regex never
           matches;
        2. ``dumpsys activity activities`` (0.05 s, 44 KB) — and it emits **no
           ``mResumedActivity`` line either**, so the disambiguating package
           stays ``None``;
        3. ``dumpsys activity top`` — the fallback, and the whole cost:
           **10.06 s**, ending in
           ``*** SERVICE 'activity' DUMP TIMEOUT (10000ms) EXPIRED ***``.

        Step 3 is not merely slow, it is **wrong**. The timeout truncates the
        dump to ~537 bytes (against 65-85 KB when it completes), so the focused
        app is frequently never listed at all; ``adbutils`` then returns the
        *last* ``ACTIVITY`` line it managed to see. Observed directly: with
        ``com.vigil.clock`` focused the truncated dump contained only market and
        chat, and ``app_current()`` confidently answered ``com.vigil.chat``. The
        trigger is ordinary exploration state — a stale app process whose
        activity dump blocks — so this is not an exotic edge case.

        ``dumpsys window displays`` (~0.04 s) carries the same information on
        its ``mFocusedApp=ActivityRecord{...}`` line and is not a per-process
        dump, so it cannot time out this way. If a future adbutils or platform
        release restores ``mCurrentFocus``, this helper becomes redundant but
        stays correct — it does not need to be reverted.

        Accuracy, not only speed: this makes ``_capture_screen``'s foreign-package
        guard, the scope classification, LEFT_APP detection, and
        :meth:`_wait_for_app_foreground` report the actually-focused app instead
        of an arbitrary survivor of a truncated dump.

        Fall back, never fail: a raising shell call or a regex miss (e.g.
        ``mFocusedApp=null`` while nothing is focused) logs at debug and
        degrades to the original ``self._device.app_current()`` path. That
        includes degrading to its *exception* behavior — when the fallback
        itself raises, the exception propagates, which is exactly what every
        call site already handles today (all five wrap the foreground read in a
        ``try``, and each has its own meaning for a failed read: ``None`` for a
        capture or a settle probe, OUT_OF_SCOPE_EXTERNAL for a live scope
        query, ``""`` for a package query). Swallowing it here would instead
        hand callers an empty package, and :meth:`_capture_screen` would then
        register a screen with ``activity_name=None`` — a different, spurious
        ``state_id`` for a screen whose activity is perfectly knowable.

        Returns:
            ``(package, activity)``, both possibly empty strings when the
            device reports no focused app.
        """
        assert self._device is not None
        try:
            raw = self._shell_output("dumpsys window displays")
        except Exception:
            logger.debug(
                "foreground: 'dumpsys window displays' raised; falling back to app_current()",
                exc_info=True,
            )
        else:
            match = FOCUSED_APP_RE.search(raw)
            if match is not None:
                return match.group("package").strip(), match.group("activity").strip()
            logger.debug("foreground: no mFocusedApp match; falling back to app_current()")
        current = self._device.app_current()
        return (
            (current.get("package", "") or "").strip(),
            (current.get("activity", "") or "").strip(),
        )

    def _disable_animation_scales(self) -> None:
        """Zero the three global animation scales for the duration of the run.

        Animated transitions are the main source of mid-render hierarchy dumps,
        which silently mint phantom states (state identity is settled-XML only,
        and there is no detector for an unsettled dump). Zeroing the scales is a
        capture-quality / enabler change; the direct wall-clock saving is small
        (roughly 0-0.2 s per step).

        Crash ratchet: the prior values are only restored by ``explore``'s
        ``finally``. If a previous run was killed before it could restore, this
        run reads a prior of ``0`` and later "restores" ``0`` — harmless on a
        dedicated exploration emulator, but it does leave animations disabled
        for any other tooling sharing that device.

        Failures are logged and swallowed: a device that refuses settings writes
        must keep exploring at today's behavior.
        """
        assert self._device is not None
        prior: dict[str, str | None] = {}
        try:
            for key in self.ANIMATION_SCALE_KEYS:
                raw = self._shell_output(f"settings get global {key}")
                try:
                    prior[key] = str(float(raw))
                except ValueError:
                    # "null" / empty / unparseable → treat as unset. Restore
                    # then deletes the key so the platform default applies,
                    # rather than writing a bogus literal back.
                    prior[key] = None
        except Exception:
            logger.warning(
                "Could not read animation scales; continuing with device defaults",
                exc_info=True,
            )
            return
        # Arm the restore before writing, so a write that fails halfway is
        # still undone by the ``finally``.
        self._animation_scales_prior = prior
        self._animation_scales_overridden = True
        try:
            for key in self.ANIMATION_SCALE_KEYS:
                self._shell_output(f"settings put global {key} 0.0")
        except Exception:
            logger.warning(
                "Could not disable animation scales; continuing with device defaults",
                exc_info=True,
            )
            return
        logger.info(f"Animation scales set to 0 (prior: {prior})")

    def _restore_animation_scales(self) -> None:
        """Restore the pre-run animation scales. Log-only on failure.

        Runs from ``explore``'s ``finally``, i.e. after the trace file is
        written, so a dead emulator at teardown can never mask the primary
        exception or lose the run's evidence.
        """
        if not self._animation_scales_overridden:
            return
        try:
            for key, value in self._animation_scales_prior.items():
                if value is None:
                    self._shell_output(f"settings delete global {key}")
                else:
                    self._shell_output(f"settings put global {key} {value}")
        except Exception:
            logger.warning("Could not restore animation scales", exc_info=True)
            return
        self._animation_scales_overridden = False
        logger.info("Animation scales restored")

    def _apply_wait_for_idle_cap(self) -> None:
        """Clamp the uiautomator server's global wait-for-idle timeout.

        ``dump_hierarchy`` goes through jsonrpc, whose client-side HTTP timeout
        is 10 s (``uiautomator2.core._jsonrpc_call``). On an animating screen a
        server-side ``waitForIdle`` of the same order races that timeout, and
        whenever the HTTP side loses, ``core.jsonrpc_call`` catches the error
        and performs a full ``stop_uiautomator`` + ``start_uiautomator`` +
        retry — a multi-second restart cascade on exactly the screens that were
        already slow. Bounding the server-side wait keeps the dump inside the
        HTTP budget so the restart path is never entered for this reason.

        Clamp semantics, deliberately one-directional: the current value is read
        back first and the cap is written only when the server is configured to
        wait *longer* than ``WAIT_FOR_IDLE_CAP_MS``. Measured on the project
        emulator (uiautomator2 3.5.0, apk 2.4.0), a freshly started server
        reports ``waitForIdleTimeout=0`` — raising that to the cap would add
        idle waiting that the run does not pay today, so an already-lower value
        is left alone. On a server whose default is the framework's 10 s, this
        writes the bound. If the read itself fails the current value is unknown,
        so nothing is written at all: a blind write is exactly what could raise a
        server that was already faster, which is the one thing the clamp promises
        never to do.

        Partial-update safety: verified against ``emulator-5554`` that
        ``setConfigurator`` with only ``waitForIdleTimeout`` merges — the
        response and a follow-up ``getConfigurator`` both still reported
        ``actionAcknowledgmentTimeout=500``, ``scrollAcknowledgmentTimeout=200``,
        ``keyInjectionDelay=0``, ``waitForSelectorTimeout=0``, ``uiAutomationFlags=0``
        unchanged. So no other timeout is zeroed and none need to be restated.

        Must be re-applied after anything that can restart the uiautomator
        server, because the server re-initialises its ``Configurator`` on
        startup and silently drops the cap. Failures are logged and swallowed:
        a device that refuses the call keeps exploring at today's behavior.
        """
        assert self._device is not None
        current: int | None
        try:
            current = int(self._device.jsonrpc.getConfigurator()["waitForIdleTimeout"])
        except Exception:
            # Unreadable configurator (older server, transient jsonrpc error, or
            # a stubbed device). The current value is unknown, so a write here
            # could *raise* a server that already waits less than the cap —
            # the clamp is one-directional, so leave the server untouched.
            logger.debug("getConfigurator failed; leaving wait-for-idle alone", exc_info=True)
            return
        if current <= self.WAIT_FOR_IDLE_CAP_MS:
            logger.debug(f"wait-for-idle already {current}ms; leaving it alone")
            return
        try:
            self._device.jsonrpc.setConfigurator({"waitForIdleTimeout": self.WAIT_FOR_IDLE_CAP_MS})
        except Exception:
            logger.warning(
                "Could not cap the uiautomator wait-for-idle timeout; "
                "continuing with the server default",
                exc_info=True,
            )
            return
        logger.info(f"wait-for-idle capped at {self.WAIT_FOR_IDLE_CAP_MS}ms (was {current})")

    def _probe_settle_observable(self) -> tuple[str, str] | None:
        """One throwaway hierarchy dump reduced to the settle observable.

        Returns:
            ``(structural_fingerprint, page_title)`` — the two content inputs of
            :meth:`RawScreen.get_hybrid_state_id`, i.e. exactly what a recorded
            capture would mint its state identity from. ``None`` when the probe
            cannot be trusted: the dump raised, the parse yielded no elements,
            or a foreign package holds the foreground. Callers MUST treat
            ``None`` as *unstable*, never as agreement.

        The observable is the structural fingerprint, not
        ``get_functional_state_key``'s anchor set — chosen because the
        fingerprint is *what identity is minted from*, so the settle checks the
        same bytes the recorded capture will be judged on. Anchors are not: they
        cover only interactable / title / toolbar text, so two probes could
        agree on stable header anchors while the fingerprint underneath was
        still churning, the settle would declare stability at the floor, and the
        capture that followed would carry a transient fingerprint — silently
        minting a phantom state. Ticking text is safe here for the opposite
        reason: text is deliberately excluded from the structural fingerprint.

        Honest scope limits — the fingerprint does NOT see everything:

        * It excludes the descendants of scrollable containers (``models/
          state.py``), so churn *inside* a list — a fling still decelerating,
          rows streaming in, a skeleton placeholder that lives in the list — is
          invisible to this observable. That is exactly why a swipe gets the
          blind :attr:`POST_SWIPE_WAIT` instead of the settle.
        * It excludes text, ``content-description``, checked state and bounds,
          so a spinner or progress indicator that only *animates* — without
          adding or removing nodes — reads as settled. Only churn that changes
          the (class, resource-id, depth, interactability) multiset outside
          scrollable subtrees is detected.

        Zeroed animation scales (:meth:`_disable_animation_scales`) are what
        keep the remaining blind spot small; the settle is a strict improvement
        on the blind sleep it replaced, not a proof of quiescence.

        Dump-only by construction: no screenshot, no XML persisted, and no
        ``scr_XXXX`` id consumed, so a probe can never enter the ``screens``
        dict (the evidence boundary) and can never shift the numbering that
        registered captures share with their on-disk files.
        """
        assert self._device is not None
        try:
            package_name, activity_name = self._foreground_app()
            if package_name and package_name != self._app_package:
                return None
            xml_string = self._device.dump_hierarchy()
        except Exception:
            logger.debug("settle probe failed", exc_info=True)
            return None
        elements = parse_hierarchy_xml(xml_string, app_package=self._app_package)
        if not elements:
            return None
        probe = RawScreen(
            screen_id="",
            activity_name=activity_name or None,
            package_name=package_name or None,
            elements=elements,
        )
        return probe.get_structural_fingerprint(), probe.extract_page_title()

    def _wait_until_settled(self, *, cap: float, registered_cap: bool = False) -> bool:
        """Wait for the screen to stop changing structurally, bounded by ``cap``.

        Replaces the blind ``time.sleep(POST_ACTION_WAIT)`` that used to follow
        every action and nav-replay step. Sleeps a short floor, probes, then
        re-probes every :attr:`SETTLE_INTERVAL`; two consecutive probes with the
        same observable mean the screen is settled and the caller may take its
        real capture.

        NOT a debounce for scrolling — see :attr:`POST_SWIPE_WAIT`. Callers that
        may have just swiped must go through :meth:`_wait_after_action` (or sleep
        :attr:`POST_SWIPE_WAIT` themselves) first.

        Args:
            cap: Wall-clock budget. Raised to :attr:`POST_ACTION_WAIT` when
                smaller, so giving up can never leave less settling than the
                fixed sleep this method replaced.
            registered_cap: True at sites that spend the
                :attr:`SETTLE_CAP_REGISTERED` budget because the capture that
                *may* follow them is registered into the result's ``screens``
                dict. Only affects the cap-hit counters, and it counts the
                sites, not the registered captures — the following capture can
                be skipped entirely (the post-action scope check bailing to
                LEFT_APP) or be transient (a cold start feeding an enumeration
                sweep), so the counter is an upper bound.

        Returns:
            True when two consecutive probes agreed; False on a cap hit.

        This is quality machinery, not a speed-up: two dump-and-parse cycles
        cost roughly what the 1.0 s sleep did. The wall-clock win comes from
        :meth:`_foreground_app` (plus the cold-start and warm-enumeration
        changes); :meth:`_apply_wait_for_idle_cap` is inert on emulators whose
        server already reports ``waitForIdleTimeout`` at or below the cap, which
        is the case on the Pixel_6a API 34 image measured here.

        A probe that returns ``None`` — dump raised, empty parse, or a foreign
        foreground package — counts as unstable and the loop keeps going to the
        cap. Returning early on those would let the post-action scope check fire
        during a transient launcher or system frame, which is how a spurious
        ``LEFT_APP`` sentinel (and a permanently blacklisted action) gets
        recorded.
        """
        deadline = time.monotonic() + max(cap, self.POST_ACTION_WAIT)
        time.sleep(self.SETTLE_FLOOR)
        last = self._probe_settle_observable()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(self.SETTLE_INTERVAL, remaining))
            observed = self._probe_settle_observable()
            if observed is not None and observed == last:
                return True
            last = observed
        self._settle_cap_hits += 1
        if registered_cap:
            self._settle_cap_hits_registered_cap += 1
        logger.debug(f"settle gave up after {cap:.1f}s (registered_cap={registered_cap})")
        return False

    def _wait_after_action(
        self, action: Action, *, cap: float, registered_cap: bool = False
    ) -> None:
        """Debounce after executing ``action``, then settle.

        A SCROLL action gets the blind :attr:`POST_SWIPE_WAIT` *first*. The
        settle observable is fingerprint + page title, and the structural
        fingerprint drops the descendants of scrollable subtrees, so a fling is
        invisible to it: two probes agree at the floor and the settle returns
        while the list is still moving. Every other action type is structurally
        visible, so the settle alone is the right (and adaptive) wait.

        This is why the debounce keys off the action type rather than a "did we
        swipe" flag: the swipes inside ``_resolve_action_target``'s
        scroll-to-find loop are already debounced at that loop's own call site,
        and they finish before the action under observation is executed.
        """
        if action.action_type in (ActionType.SCROLL_UP, ActionType.SCROLL_DOWN):
            time.sleep(self.POST_SWIPE_WAIT)
        self._wait_until_settled(cap=cap, registered_cap=registered_cap)

    def _cold_start_app(self) -> bool:
        """home → ``app_start(stop=True)`` → wait for foreground → settle.

        If the initial attempt doesn't land the target app in the foreground
        within ``WAIT_FOR_FOREGROUND`` seconds, run one recovery pass: if a
        foreign app is sitting in the foreground (emulator sluggishness
        from Files / Gallery / etc. launched in a prior step), force-stop
        it, sleep 3 s, then retry the home→start sequence once more.
        This absorbs transient emulator contention that otherwise triggers
        premature COLD_START_FAILED cascades.

        The wait-for-idle cap is re-applied by :meth:`_cold_start_attempt`
        itself, *before* its settle — that settle is the first thing to dump the
        freshly launched, still-animating screen, so re-applying only after the
        attempt returned would leave exactly those dumps running against the
        uncapped server timeout.
        """
        assert self._device is not None
        if self._cold_start_attempt():
            return True

        foreground_pkg = ""
        try:
            foreground_pkg = self._foreground_app()[0]
        except Exception:
            logger.debug("cold_start recovery: foreground query raised", exc_info=True)

        if foreground_pkg and foreground_pkg != self._app_package:
            logger.warning(
                f"cold_start recovery: foreground is {foreground_pkg!r}, "
                "force-stopping and retrying"
            )
            try:
                self._device.app_stop(foreground_pkg)
            except Exception:
                logger.debug(f"force-stop {foreground_pkg} raised", exc_info=True)
        else:
            logger.warning(
                "cold_start recovery: no foreign foreground detected, "
                "waiting 3s for emulator to settle and retrying"
            )
        time.sleep(3.0)
        return self._cold_start_attempt()

    def _cold_start_attempt(self) -> bool:
        """One home → ``app_start(stop=True)`` → foreground-poll → settle cycle.

        The force-stop is ``app_start(pkg, stop=True)``'s own — uiautomator2
        force-stops the package on the ``stop`` branch before launching — so
        this method issues no separate ``app_stop`` call and no sleep for one.
        (The ``app_stop`` in :meth:`_cold_start_app` is a different thing: it
        targets a *foreign* foreground package on the recovery pass.)

        There is no blind post-launch sleep. The launch is detected by
        :meth:`_wait_for_app_foreground`, then the screen is settled adaptively
        before returning, because :meth:`_foreground_app` flips to the target
        package at activity-resume time — *before* the first frame — and the entry-state
        capture that follows a successful cold start is registered evidence with
        no selector gate and no drift check on the ``needs_nav`` path. An
        unsettled dump there would silently mint a phantom state.

        An ``app_start`` failure is fatal to the attempt and returns False
        without polling. Nothing else in this method kills the old instance, so
        a raised ``app_start`` can leave the previous process still resumed —
        polling on would let it satisfy the foreground check and make a failed
        launch look like a successful cold start. The home press stays
        best-effort: it is only a debounce for the poll.

        The wait-for-idle cap is re-applied here, after the foreground poll and
        *before* the settle. The ``app_start`` force-stop / relaunch churn is one
        of the things that leaves the uiautomator server restarted
        (``core.jsonrpc_call`` restarts it on any HTTP or UiAutomation error) and
        a restarted server silently resets its ``Configurator``; the settle that
        follows then issues several ``dump_hierarchy`` calls against a freshly
        launched, still-animating screen — the exact stall the cap exists to
        prevent, on the screen most likely to stall. Applying it afterwards would
        be too late for those dumps.
        """
        assert self._device is not None
        dev, pkg = self._device, self._app_package
        try:
            dev.press("home")
        except Exception:
            logger.debug("cold_start.home raised", exc_info=True)
        time.sleep(self.COLD_START_HOME_WAIT)
        try:
            dev.app_start(pkg, stop=True)
        except Exception:
            logger.debug("cold_start.app_start raised", exc_info=True)
            return False
        if not self._wait_for_app_foreground():
            return False
        self._apply_wait_for_idle_cap()
        # Known-fresh viewport: the app was force-stopped and relaunched, so
        # every scrollable container is back at its default offset. This is the
        # one unconditionally sound reset for the warm-enumeration gate.
        self._device_scrolled = False
        self._wait_until_settled(cap=self.SETTLE_CAP_REGISTERED, registered_cap=True)
        return True

    # ------------------------- prior-guided intent launch ------------------

    def _capture_screen(self, *, take_screenshot: bool = True) -> RawScreen | None:
        """Dump the live hierarchy into a :class:`RawScreen`.

        Args:
            take_screenshot: When True (the default) a PNG is pulled from the
                device. Pass False only at *transient* capture sites — those
                whose :class:`RawScreen` is used for selector resolution or
                scroll bookkeeping and is never registered into the result's
                ``screens`` dict. The screens dict is the evidence boundary:
                every screenshot consumer resolves images exclusively through
                it, so a screenshot taken for a capture that can never be
                registered is unreachable by any consumer. Registered captures
                (entry, ``pre``, drift-fallback ``pre``, ``post``) keep the
                default so their on-disk PNG exists. The default is True so
                that any future call site is safe without an audit.

        The screen counter, ``screenshot_path`` field, XML disk write, and
        element parse are identical in both modes, keeping ``scr_XXXX`` ids
        aligned with on-disk filenames and the ``RawScreen`` shape unchanged.
        Transient XML is still persisted so resolution failures stay
        debuggable offline.

        Device-read order is :meth:`_foreground_app` → screenshot →
        ``dump_hierarchy``,
        i.e. **the dump is deliberately the last device read**. The caller of a
        registered ``pre`` capture resolves its click target against this
        screen's element bounds, so every device round-trip that sits between
        the dump and the click ages those bounds by its own latency. Pulling the
        PNG (the slowest read here) before the dump instead of after removes the
        whole PNG transfer from that dump-to-click gap. It buys no wall-clock
        time — the same three reads happen — it only moves the freshness.

        Counter discipline is unchanged by that reorder: the id is *reserved*
        (``_screen_counter + 1``) for the PNG filename and only *committed* once
        the dump returns, so a raising dump still consumes no ``scr_XXXX`` id.
        The next capture reuses the id and overwrites the orphaned PNG, exactly
        as if the screenshot had never been taken. ``screenshot_path`` still
        points at ``screens/<screen_id>.png`` for the id this capture returns.
        """
        assert self._device is not None
        try:
            package_name, activity_name = self._foreground_app()
            if package_name and package_name != self._app_package:
                return None
            # Reserved, not committed: see the docstring's counter discipline.
            screen_id = f"scr_{self._screen_counter + 1:04d}"
            screenshot_path = self._output_dir / "screens" / f"{screen_id}.png"
            if take_screenshot:
                try:
                    self._device.screenshot(str(screenshot_path))
                except Exception:
                    logger.debug(f"screenshot failed for {screen_id}", exc_info=True)
            xml_string = self._device.dump_hierarchy()
        except Exception:
            logger.exception("dump_hierarchy failed")
            return None
        self._screen_counter += 1
        xml_path = self._output_dir / "trees" / f"{screen_id}.xml"
        try:
            xml_path.write_text(xml_string, encoding="utf-8")
        except Exception:
            logger.debug(f"xml persist failed for {screen_id}", exc_info=True)
        elements = parse_hierarchy_xml(xml_string, app_package=self._app_package)
        if not elements:
            return None
        scr = RawScreen(
            screen_id=screen_id,
            activity_name=activity_name or None,
            package_name=package_name or None,
            screenshot_path=str(screenshot_path),
            xml_tree_path=str(xml_path),
            elements=elements,
            timestamp=_now_iso(),
            metadata={"device_serial": self._serial},
        )
        # Stash the extracted toolbar title in metadata so FSM builder and
        # other downstream passes can read it without re-walking the tree.
        scr.metadata["page_title"] = scr.extract_page_title()
        return scr

    def _execute_action(self, action: Action, *, screen: RawScreen | None = None) -> bool:
        """Dispatch. Coordinates are resolved live — stored bounds are a hint.

        Args:
            action: The action to perform.
            screen: Optional already-captured screen to resolve the target
                against. Pass the observation's ``pre`` capture — the screen
                the emitted trace records as ``source_screen_id`` — so the
                element that gets clicked comes from the recorded evidence.
                Only safe when *no device input* happened between that capture
                and this call. Defaults to None (capture fresh), which is what
                replay call sites must keep using: the device moves at every
                replay step, so a reused screen there would be genuinely stale.

        Side channel: writes diagnostics for the most recent execution into
        ``self._last_execution_metadata`` so :meth:`_perform_one_observation`
        can attach them to the emitted :class:`ExplorationTrace`. Keys may
        include ``selector_resolution`` (match / missing / ambiguous),
        ``cleared``, ``input_policy``.

        Strict resolution: for CLICK / LONG_PRESS / INPUT_TEXT, if the
        action carries a stable selector we use three-valued
        :func:`vigil.neuro.selector_resolution.resolve_selector`. AMBIGUOUS
        results return False without firing a click — the explorer never
        guesses which of N matching widgets to interact with.
        """
        assert self._device is not None
        self._last_execution_metadata = {}
        try:
            t = action.action_type
            if t == ActionType.NAVIGATE_BACK:
                self._device.press("back")
                return True
            if t == ActionType.NAVIGATE_HOME:
                self._device.press("home")
                return True
            if t in (ActionType.SCROLL_UP, ActionType.SCROLL_DOWN):
                return self._execute_scroll(action, screen=screen)
            if t in (ActionType.CLICK, ActionType.LONG_PRESS, ActionType.INPUT_TEXT):
                element, resolution_status, resolved_screen = self._strict_resolve_target(
                    action, screen=screen
                )
                # setdefault, not assignment: `_strict_resolve_target` writes
                # "legacy" itself when it matched via the descriptor fallback,
                # and that value is part of the load-bearing
                # match / missing / ambiguous / legacy vocabulary that
                # validate_fsm reads out of trace metadata. Overwriting it with
                # the ResolutionStatus ("match") meant "legacy" never reached a
                # single trace. Nothing else can have populated the key: the
                # metadata dict is reset at the top of this method.
                self._last_execution_metadata.setdefault(
                    "selector_resolution", resolution_status.value
                )
                if element is None:
                    logger.debug(
                        f"resolve {resolution_status.value} for {t.value} {_short_target(action)}"
                    )
                    return False
                if not element.bounds:
                    self._last_execution_metadata.setdefault(
                        "execution_error", "resolved element has no bounds"
                    )
                    return False
                cx = (element.bounds[0] + element.bounds[2]) // 2
                cy = (element.bounds[1] + element.bounds[3]) // 2
                if t == ActionType.CLICK:
                    self._device.click(cx, cy)
                    return True
                if t == ActionType.LONG_PRESS:
                    self._device.long_click(cx, cy)
                    return True
                # INPUT_TEXT: clear-then-set, never append. Bookkeeping is
                # stored in self._last_execution_metadata so the enclosing
                # observation can record original text / cleared flag.
                return self._apply_input_text(element, action, cx, cy, resolved_screen)
        except Exception:
            logger.exception(f"execute_action({action.action_type.value}) raised")
            return False
        return False

    def _strict_resolve_target(
        self, action: Action, *, screen: RawScreen | None = None
    ) -> tuple[UIElement | None, ResolutionStatus, RawScreen | None]:
        """Resolve a CLICK / LONG_PRESS / INPUT_TEXT target with three-valued
        result. AMBIGUOUS resolutions return ``(None, AMBIGUOUS)`` so the
        explorer can refuse to execute. Falls back to legacy descriptor
        matching only when the action carries no stable selector — and in
        that legacy case the descriptor matcher is the (older) first-match
        path; the trace metadata records ``selector_resolution="legacy"``.

        Args:
            action: The action whose target must be located.
            screen: Optional caller-supplied screen (the observation's ``pre``)
                to resolve against instead of taking a fresh dump. A MISS on a
                reused screen always triggers exactly one fresh capture and a
                re-match *before* any scroll-to-find, so an element that
                appeared asynchronously can never turn a clean observation into
                a scroll-contaminated one. AMBIGUOUS is never retried on a
                fresh capture: refusing on the recorded evidence screen is what
                keeps the explorer's refusal consistent with validate_fsm.

        Which screen produced the status: AMBIGUOUS and every MATCH found on the
        reused screen come from the recorded evidence screen. The MISS-then-
        recapture path above is the one exception — from there on the status
        (and any scroll-to-find that follows) is decided on a screen the trace
        does *not* record as ``source_screen_id``. That divergence is not
        silent: this method writes ``selector_resolution_recaptured=True`` and
        ``selector_resolution_screen_id`` into
        ``self._last_execution_metadata``, so post-hoc analysis can tell a
        status read off the recorded evidence from one read off a later dump.
        """
        reused = screen is not None
        if screen is None:
            screen = self._capture_screen(take_screenshot=False)
        if screen is None:
            return None, ResolutionStatus.MISSING, None

        if selector_has_stable_identity(action.target_selector):
            result = resolve_selector(
                action.target_selector, screen.elements, action_type=action.action_type
            )
            if result.status == ResolutionStatus.MISSING and reused:
                # Never enter the device-mutating scroll loop on a miss seen
                # only on the reused screen. Re-observe once, first.
                fresh = self._capture_screen(take_screenshot=False)
                if fresh is None:
                    return None, ResolutionStatus.MISSING, screen
                screen = fresh
                self._note_resolution_recapture(screen)
                result = resolve_selector(
                    action.target_selector, screen.elements, action_type=action.action_type
                )
            if result.status == ResolutionStatus.MATCH:
                return result.element, ResolutionStatus.MATCH, screen
            if result.status == ResolutionStatus.AMBIGUOUS:
                self._last_execution_metadata.setdefault(
                    "selector_resolution_reason", result.reason
                )
                self._last_execution_metadata["selector_candidates"] = result.candidates
                return None, ResolutionStatus.AMBIGUOUS, screen
            # MISSING from selector → try scroll-to-find as before, seeded with
            # the freshly observed screen so the resolver does not re-dump it.
            scrolled, screen_after = self._resolve_action_target(
                action, scroll_to_find=True, screen=screen
            )
            if scrolled is not None:
                # Confirm uniqueness on the screen the element was found on.
                if screen_after is not None:
                    recheck = resolve_selector(
                        action.target_selector,
                        screen_after.elements,
                        action_type=action.action_type,
                    )
                    if recheck.status == ResolutionStatus.AMBIGUOUS:
                        self._last_execution_metadata.setdefault(
                            "selector_resolution_reason", recheck.reason
                        )
                        return None, ResolutionStatus.AMBIGUOUS, screen_after
                    if recheck.status == ResolutionStatus.MATCH:
                        return recheck.element, ResolutionStatus.MATCH, screen_after
                return scrolled, ResolutionStatus.MATCH, screen_after
            return None, ResolutionStatus.MISSING, screen

        # No stable selector — fall back to the legacy descriptor matcher.
        if reused:
            # Same guardrail as above: match on the reused screen without
            # scrolling; only after one fresh re-observation may we scroll.
            found = self._match_descriptor(screen, action)
            if found is not None:
                self._last_execution_metadata["selector_resolution"] = "legacy"
                return found, ResolutionStatus.MATCH, screen
            fresh = self._capture_screen(take_screenshot=False)
            if fresh is None:
                return None, ResolutionStatus.MISSING, screen
            screen = fresh
            self._note_resolution_recapture(screen)
        legacy, legacy_screen = self._resolve_action_target(
            action, scroll_to_find=True, screen=screen
        )
        if legacy is None:
            return None, ResolutionStatus.MISSING, legacy_screen or screen
        self._last_execution_metadata["selector_resolution"] = "legacy"
        return legacy, ResolutionStatus.MATCH, legacy_screen or screen

    def _note_resolution_recapture(self, screen: RawScreen) -> None:
        """Flag that resolution left the recorded evidence screen.

        Called only from the MISS-on-reused-screen re-observation in
        :meth:`_strict_resolve_target`. From that point the final
        ``selector_resolution`` status is decided on ``screen``, which is a
        transient capture and therefore *not* the ``source_screen_id`` the
        emitted trace records. The two keys land in the trace's execution
        metadata (via ``_last_execution_metadata``) so a post-hoc reader can
        separate a status read off the recorded evidence from one read off a
        later dump. They are additive diagnostics: the load-bearing
        ``selector_resolution`` vocabulary is untouched.
        """
        self._last_execution_metadata["selector_resolution_recaptured"] = True
        self._last_execution_metadata["selector_resolution_screen_id"] = screen.screen_id

    def _apply_input_text(
        self,
        element: UIElement,
        action: Action,
        cx: int,
        cy: int,
        current_screen: RawScreen | None,
    ) -> bool:
        """Safely apply INPUT_TEXT with a strict clear-before-set contract.

        The explorer NEVER appends text. If the device cannot atomically
        clear the existing text before setting (uiautomator2 element-level
        ``set_text`` is preferred; ``clear_text`` is a fallback), this
        method returns False and records ``cleared=False`` /
        ``input_policy="skipped_unsafe"`` in metadata so the enclosing
        observation emits ``SENTINEL_ACTION_FAILED`` rather than mutating
        user data with an append.
        """
        assert self._device is not None
        cleared = False

        selector_kwargs = self._input_text_selector_kwargs(element, current_screen)

        # 1. Preferred path: element-level set_text via a selector that is
        #    unique for the resolved element.
        try:
            selector: Any = None
            if selector_kwargs is not None:
                selector = self._device(**selector_kwargs)
            elif element.class_name and not self._last_execution_metadata.get(
                "input_selector_collision"
            ):
                selector = self._device(className=element.class_name)
            if selector is not None:
                try:
                    exists = bool(selector.exists)
                except Exception:
                    exists = True
                if exists:
                    selector.set_text(action.input_text or "")
                    cleared = True
                    self._last_execution_metadata["input_policy"] = "clear_set"
        except Exception:
            logger.debug("element.set_text path failed", exc_info=True)

        if not cleared:
            # 2. Fallback: focus, then clear_text. Only if clear succeeds
            #    do we proceed to send_keys. Otherwise refuse.
            try:
                self._device.click(cx, cy)
                time.sleep(0.3)
            except Exception:
                logger.debug("click for focus before INPUT_TEXT failed", exc_info=True)
            try:
                self._device.clear_text()
                cleared = True
            except Exception:
                logger.debug("device.clear_text failed", exc_info=True)
            if cleared:
                try:
                    if action.input_text:
                        self._device.send_keys(action.input_text)
                    self._last_execution_metadata["input_policy"] = "clear_set"
                except Exception:
                    logger.debug("send_keys after clear failed", exc_info=True)
                    self._last_execution_metadata["cleared"] = True
                    self._last_execution_metadata["input_policy"] = "send_failed"
                    return False

        if not cleared:
            # Cannot guarantee clear-before-set: refuse rather than append.
            self._last_execution_metadata["cleared"] = False
            self._last_execution_metadata["input_policy"] = "skipped_unsafe"
            logger.warning(
                "INPUT_TEXT refused: cannot clear existing text without "
                "possibly appending to the existing value"
            )
            return False

        try:
            self._device.hide_keyboard()
        except Exception:
            logger.debug("keyboard dismiss failed", exc_info=True)
        time.sleep(0.3)

        self._last_execution_metadata["cleared"] = True
        return True

    def _input_text_selector_kwargs(
        self, element: UIElement, current_screen: RawScreen | None
    ) -> dict[str, Any] | None:
        """Build a selector that targets the resolved EditText identity."""
        if not element.resource_id:
            return None

        selector_kwargs: dict[str, Any] = {"resourceId": element.resource_id}
        if not current_screen:
            self._last_execution_metadata["input_selector_collision"] = True
            return None

        editable_same_rid = [
            e
            for e in current_screen.elements
            if e.is_editable and e.resource_id == element.resource_id
        ]
        if len(editable_same_rid) <= 1:
            return selector_kwargs

        indexed_peers = editable_same_rid
        if element.class_name:
            selector_kwargs["className"] = element.class_name
            indexed_peers = [e for e in editable_same_rid if e.class_name == element.class_name]

        target_index = self._resolved_element_index(indexed_peers, element)
        self._last_execution_metadata["input_selector_collision"] = True
        if target_index is None:
            return None

        selector_kwargs["instance"] = target_index
        return selector_kwargs

    @staticmethod
    def _resolved_element_index(elements: list[UIElement], target: UIElement) -> int | None:
        for index, candidate in enumerate(elements):
            if candidate is target:
                return index
            if candidate.element_id and candidate.element_id == target.element_id:
                return index
            if candidate.bounds == target.bounds and candidate.class_name == target.class_name:
                return index
        return None

    def _match_descriptor(self, screen: RawScreen, action: Action) -> UIElement | None:
        """Find the element whose descriptor matches ``action``.

        Resolution order:
          1. If the action carries a ``target_selector`` with stable identity,
             try :func:`vigil.core.ui_selectors.find_element_by_selector`.
          2. Fall back to descriptor matching on (rid, text, content-desc,
             class) when the selector misses on the live screen or the
             action carries no stable selector.

        Descriptor matching rules (all specified fields must be compatible):
          1. If ``rid`` is specified in the action, the element's rid must match.
          2. If ``class_name`` is specified, must match.
          3. If ``text`` is specified:
               - match if element's own normalized text equals it, OR
               - (Preference-row fallback) the element has a direct child
                 with ``resource_id == "android:id/title"`` whose normalized
                 text matches.
          4. If ``content_desc`` is specified, the element's content_desc
             must match.
          5. If only ``rid`` is specified (no text / cd), any rid match wins.
          6. If none of ``rid`` / ``text`` / ``cd`` / ``cls`` is specified,
             no match (caller should have rejected the action upstream).

        Returns the first matching element, or None.
        """
        if selector_has_stable_identity(action.target_selector):
            found = find_element_by_selector(action.target_selector, screen.elements)
            if found is not None:
                return found

        rid = action.target_resource_id or ""
        text = _normalize_dynamic(action.target_text or "") if action.target_text else ""
        cd = action.target_content_desc or ""
        cls = action.target_class_name or ""
        if not (rid or text or cd or cls):
            return None

        by_id = {e.element_id: e for e in screen.elements}

        def title_in_subtree(e: UIElement) -> bool:
            """True when an ``android:id/title`` descendant's normalized text
            equals the target — the Preference-row pattern where the title
            TextView lives two or three layouts down from the clickable
            parent."""
            return bool(text) and text in _descendant_title_texts(e, by_id)

        for e in screen.elements:
            e_rid = e.resource_id or ""
            e_text = _normalize_dynamic(e.text or "") if e.text else ""
            e_cd = e.content_description or ""
            e_cls = e.class_name or ""

            if rid and rid != e_rid:
                continue
            if cls and cls != e_cls:
                continue
            if cd and e_cd and cd != e_cd:
                continue
            if text:
                if e_text == text:
                    return e
                if not e_text and title_in_subtree(e):
                    return e
                continue
            if cd:
                if e_cd == cd:
                    return e
                continue
            if rid:
                # rid+class (or rid alone) with no text/cd specified.
                return e
            if cls and e_cls == cls and not text and not cd:
                return e
        return None

    def _resolve_action_target(
        self, action: Action, *, scroll_to_find: bool, screen: RawScreen | None = None
    ) -> tuple[UIElement | None, RawScreen | None]:
        """Find the descriptor target, scrolling to reveal it if necessary.

        Args:
            action: The action whose descriptor identifies the target.
            scroll_to_find: Whether to swipe looking for an off-screen target.
            screen: Optional screen to start from. When None (the default) the
                current screen is captured. Callers that have *just* observed
                the live screen pass it here so the identical dump is not taken
                twice.

        Returns:
            ``(element, screen)`` where ``screen`` is the capture the element
            was matched on — the caller can re-resolve against it instead of
            taking another dump. ``element`` is None when the target could not
            be found; ``screen`` is then the last capture examined, or None if
            capturing failed outright.
        """
        if screen is None:
            screen = self._capture_screen(take_screenshot=False)
        if screen is None:
            return None, None
        found = self._match_descriptor(screen, action)
        if found is not None:
            return found, screen
        if not scroll_to_find:
            return None, screen

        last_anchors: frozenset[tuple[str, str]] | None = None
        for _ in range(self.MAX_SCROLL_TO_FIND):
            scroll_action = _build_scroll_down(screen)
            if scroll_action is None:
                return None, screen
            _, anchors = screen.get_functional_state_key(self._app_package)
            if last_anchors is not None and anchors == last_anchors:
                return None, screen
            last_anchors = anchors
            if not self._swipe_scroll(ActionType.SCROLL_DOWN, scroll_action):
                return None, screen
            # Blind wall-clock debounce, NOT the settle: the settle cannot see a
            # fling (see POST_SWIPE_WAIT). Matching a descriptor against a
            # still-moving list mis-resolves the target — and the bounds this
            # resolver returns are what the click is aimed at.
            time.sleep(self.POST_SWIPE_WAIT)
            screen = self._capture_screen(take_screenshot=False)
            if screen is None:
                return None, None
            found = self._match_descriptor(screen, action)
            if found is not None:
                return found, screen
        return None, screen

    def _execute_scroll(self, action: Action, *, screen: RawScreen | None = None) -> bool:
        """Descriptor-based scroll: find the matching scrollable on the
        current screen, swipe within its live bounds. Never falls back to
        whole-screen bounds — that would fire on unrelated containers.

        ``screen``, when supplied, is the observation's ``pre`` capture. The
        container is resolved against it and the swipe uses its bounds, so the
        gesture fires inside the container the recorded evidence shows.

        A miss on that reused screen gets exactly one fresh re-observation
        before refusing — the same one-shot guard the selector and legacy
        resolution paths already have. It is safe here for the same reason: a
        ``_capture_screen(take_screenshot=False)`` mutates nothing on the
        device, consumes no registered evidence slot, and cannot scroll. It only
        distinguishes "this container never existed" (still a refusal:
        ACTION_FAILED, no swipe, validator-consistent) from "the container
        rendered a moment after ``pre`` was dumped", which without the guard
        would be a silently lost scroll edge.
        """
        reused = screen is not None
        if screen is None:
            screen = self._capture_screen(take_screenshot=False)
        if screen is None:
            return False
        container = self._match_descriptor(screen, action)
        if (container is None or not container.bounds) and reused:
            fresh = self._capture_screen(take_screenshot=False)
            if fresh is not None:
                screen = fresh
                container = self._match_descriptor(screen, action)
        if container is None or not container.bounds:
            return False
        live = Action(
            action_type=action.action_type,
            target_bounds=container.bounds,
        )
        return self._swipe_scroll(action.action_type, live)

    def _swipe_scroll(self, t: ActionType, action: Action) -> bool:
        """SCROLL_DOWN: finger high→low (reveal content below). SCROLL_UP: opposite.

        ``action.target_bounds`` must be populated (caller's contract;
        descriptor-resolver provides live bounds). Falls back to full
        window only when called from enumeration scroll-sweep (which
        passes the container's enumeration-time bounds).

        Single gesture choke point: every swipe this class issues goes through
        here, so this is where ``_device_scrolled`` is armed. Set before the
        gesture and regardless of its outcome — a swipe that raised may still
        have moved the viewport, and the flag is only ever allowed to be
        conservative.
        """
        assert self._device is not None
        self._device_scrolled = True
        if action.target_bounds:
            cx = (action.target_bounds[0] + action.target_bounds[2]) // 2
            top, bot = action.target_bounds[1], action.target_bounds[3]
        else:
            try:
                w, h = self._device.window_size()
            except Exception:
                w, h = self.FALLBACK_WINDOW_SIZE
            cx, top, bot = w // 2, 0, h
        y_low = int(top + (bot - top) * 0.2)
        y_high = int(top + (bot - top) * 0.8)
        if t is ActionType.SCROLL_DOWN:
            self._device.swipe(cx, y_high, cx, y_low, duration=0.3)
        else:
            self._device.swipe(cx, y_low, cx, y_high, duration=0.3)
        return True

    def _wait_for_app_foreground(self, timeout: float | None = None) -> bool:
        deadline = time.monotonic() + (timeout or self.WAIT_FOR_FOREGROUND)
        while time.monotonic() < deadline:
            if self._is_within_app():
                return True
            time.sleep(self.FOREGROUND_POLL_INTERVAL)
        return False

    def _is_within_app(self) -> bool:
        cat = self._current_scope()
        return self._scope_policy.is_allowed(cat)

    def _scope_of_screen(self, screen: RawScreen) -> ScopeCategory:
        """Classify the scope that a captured evidence screen was taken in.

        Not ``classify(screen.package_name)``. :meth:`_capture_screen` returns a
        screen only after proving the foreground is not a *foreign* package —
        its guard is ``if package_name and package_name != self._app_package``,
        so an **empty** package passes and is then stored as
        ``package_name=None``. Handing that ``None`` to
        :meth:`ScopePolicy.classify` yields OUT_OF_SCOPE_EXTERNAL, which would
        label a perfectly good in-app capture with a foreign scope.

        That is not cosmetic: ``scope_pre`` is one of the three frozen
        trace-metadata inputs to ``fsm_builder._classify_trace_evidence``, so a
        spurious external scope silently drops a valid edge from the graph. An
        empty package therefore resolves to the app's own package — the only
        thing the capture guard actually established about the foreground.
        """
        return self._scope_policy.classify(screen.package_name or self._app_package)

    def _current_scope(self) -> ScopeCategory:
        """Classify the current foreground package via the scope policy."""
        assert self._device is not None
        try:
            pkg = self._foreground_app()[0]
        except Exception:
            return ScopeCategory.OUT_OF_SCOPE_EXTERNAL
        return self._scope_policy.classify(pkg)

    def _current_foreground_package(self) -> str:
        assert self._device is not None
        try:
            return self._foreground_app()[0]
        except Exception:
            return ""

    # ---------------------------------------------------------- finalize

    def _finalize(
        self,
        traces: list[ExplorationTrace],
        screens: dict[str, RawScreen],
        nav_paths: dict[str, list[Action]],
        counts: dict[str, int],
        total_steps: int,
        duration: float,
    ) -> ExplorationResult:
        nav_stats = {
            "observations_total": total_steps,
            "states_discovered": len(nav_paths),
            "drift_count": self._drift_count,
            "verified_navigation_states": len(self._verified_navigation_states),
            "unverified_navigation_targets": len(self._unverified_navigation_targets),
            # Adaptive-settle give-ups. `settle_cap_hits_registered_cap` is the
            # one that matters for evidence quality: cap hits at the settle
            # sites that precede a *possibly* registered capture. Upper bound,
            # not an exact count of unsettled registered captures — see
            # `_wait_until_settled`.
            "settle_cap_hits": self._settle_cap_hits,
            "settle_cap_hits_registered_cap": self._settle_cap_hits_registered_cap,
            **counts,
        }
        result = ExplorationResult(
            app_package=self._app_package,
            screens=screens,
            traces=traces,
            total_steps=total_steps,
            unique_screens=len(nav_paths),
            duration_seconds=round(duration, 2),
            output_dir=str(self._output_dir),
            nav_stats=nav_stats,
        )
        self._save_result(result)
        logger.info(
            f"Done: {total_steps} observations, {len(nav_paths)} states, "
            f"{result.duration_seconds:.1f}s — saved to {result.output_dir}"
        )
        return result

    def _save_result(self, result: ExplorationResult) -> None:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        trace_path = self._output_dir / "traces" / f"exploration_{timestamp}.json"
        compact: dict[str, Any] = {}
        for sid, s in result.screens.items():
            interactable = s.get_interactable_elements()
            page_title = s.extract_page_title()
            try:
                compact_tree_text = compact_ui_tree_text(s.elements)
            except Exception:
                logger.debug(f"compact_ui_tree_text failed for {sid}", exc_info=True)
                compact_tree_text = ""
            compact[sid] = {
                "screen_id": s.screen_id,
                "activity_name": s.activity_name,
                "package_name": s.package_name,
                "screenshot_path": s.screenshot_path,
                "xml_tree_path": s.xml_tree_path,
                # ``state_id`` is the canonical identity for downstream
                # consumers (fsm_builder dedups on this). Uses the hybrid
                # key — structural fingerprint + activity + page title.
                "structural_fingerprint": s.get_structural_fingerprint(),
                "state_id": s.get_hybrid_state_id(self._app_package),
                "page_title": page_title,
                "total_elements": len(s.elements),
                # Canonical: full UIElement graph for downstream replay /
                # fingerprinting / DSL evaluation. Lossy ``compact_tree_text``
                # is for LLM prompts only — do NOT feed it to fingerprint or
                # replay code paths.
                "elements": [e.model_dump(mode="json") for e in s.elements],
                "interactable_elements": [e.model_dump(mode="json") for e in interactable],
                "compact_tree_text": compact_tree_text,
                "timestamp": s.timestamp,
                "metadata": {"page_title": page_title},
            }
        data: dict[str, Any] = {
            "app_package": result.app_package,
            "device_serial": self._serial,
            "exploration_strategy": self.STRATEGY_NAME,
            "max_steps": self._config.app.max_exploration_steps,
            "total_steps": result.total_steps,
            "unique_screens": result.unique_screens,
            "duration_seconds": result.duration_seconds,
            "timestamp": _now_iso(),
            # Environment provenance: were the global animation scales zeroed
            # for this run, and what were they before. Extra top-level keys are
            # tolerated by the artifact checker (subset match); sentinel rows,
            # the screens dict and the per-row schema are untouched.
            "animation_scales": {
                "overridden": self._animation_scales_overridden,
                "prior": dict(self._animation_scales_prior),
            },
            "screens": compact,
            "traces": [t.model_dump(mode="json") for t in result.traces],
            "action_attempts": [a.model_dump(mode="json") for a in self._action_attempts],
            "nav_stats": result.nav_stats,
        }
        trace_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(f"Exploration trace saved to {trace_path}")

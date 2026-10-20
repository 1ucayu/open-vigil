"""Three-valued selector resolution: MATCH / MISSING / AMBIGUOUS.

Existing helpers in :mod:`vigil.core.ui_selectors` return ``UIElement | None``
and silently pick the first candidate when a selector is ambiguous (e.g. a
bare ``resource_id`` that occurs on multiple widgets). Native exploration must
not click an ambiguous target — it would produce a misleading trace where the
recorded action does not match the executed widget. This module exposes a
structured result so callers can DENY ambiguous resolutions and record them in
trace metadata.

Strong identity fields: ``resource_id``, ``text``, ``content_description``,
``nearby_text``. Weak fields used only for disambiguation: ``class_name``,
``ancestor_chain``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vigil.core.ui_selectors import (  # type: ignore[attr-defined]
    _class_leaf,
    discriminative_descendant_text,
)
from vigil.models.action import ActionType
from vigil.models.state import UIElement, _normalize_dynamic, is_text_input


def _is_visible(elem: UIElement) -> bool:
    """Element has non-degenerate on-screen bounds."""
    b = elem.bounds
    if not b or len(b) < 4:
        return False
    return (b[2] - b[0]) > 0 and (b[3] - b[1]) > 0


def _action_compatible(elem: UIElement, action_type: ActionType | None) -> bool:
    """Per-action capability check.

    CLICK         -> is_clickable OR is_checkable
    LONG_PRESS    -> is_long_clickable
    INPUT_TEXT    -> is_text_input (NOT is_editable -- see below)
    SCROLL_*      -> is_scrollable
    NAVIGATE_*    -> trivially compatible (no element target)
    None          -> no constraint
    """
    if action_type is None:
        return True
    if action_type in (ActionType.NAVIGATE_BACK, ActionType.NAVIGATE_HOME):
        return True
    if action_type == ActionType.CLICK:
        return elem.is_clickable or elem.is_checkable
    if action_type == ActionType.LONG_PRESS:
        return elem.is_long_clickable
    if action_type == ActionType.INPUT_TEXT:
        # ``is_text_input``, not ``is_editable``: the parser only sets
        # ``is_editable`` for classes whose name ends in ``EditText``, which
        # excludes ``AutoCompleteTextView`` -- the class backing every Android
        # SearchView field. Gating on ``is_editable`` here would refuse to
        # resolve the very actions the explorer now enumerates. ``is_editable``
        # itself is left alone because it is part of ``state_id``.
        return is_text_input(elem)
    if action_type in (ActionType.SCROLL_UP, ActionType.SCROLL_DOWN):
        return elem.is_scrollable
    return True


def _is_candidate(elem: UIElement, action_type: ActionType | None) -> bool:
    """Combined enabled + visible + action-compatible gate."""
    if not elem.is_enabled:
        return False
    if not _is_visible(elem):
        return False
    return _action_compatible(elem, action_type)


class ResolutionStatus(StrEnum):
    """Outcome of resolving a stable selector against a captured screen."""

    MATCH = "match"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ResolutionResult:
    """Result of :func:`resolve_selector`.

    Attributes:
        status: MATCH / MISSING / AMBIGUOUS.
        element: The resolved element when status is MATCH, else None.
        candidates: Number of elements that matched the strongest available
            identity field. 1 implies MATCH, 0 implies MISSING, >1 means
            disambiguation was attempted and (if still ambiguous) failed.
        reason: Short human-readable explanation for diagnostics / trace
            metadata. Empty for MATCH.
    """

    status: ResolutionStatus
    element: UIElement | None
    candidates: int
    reason: str = ""


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return _normalize_dynamic(value)


def _matches_field(elem: UIElement, key: str, value: str) -> bool:
    if key == "resource_id":
        return (elem.resource_id or "") == value
    if key == "text":
        return _normalize_text(elem.text) == value
    if key == "content_description":
        return (elem.content_description or "") == value
    if key == "class_name":
        return elem.class_name == value or _class_leaf(elem.class_name) == _class_leaf(value)
    return False


def _matches_row_discriminator(elem: UIElement, value: str, elements: list[UIElement]) -> bool:
    """True when ``elem``'s own derived row discriminator equals ``value``.

    Deliberately symmetric with ``build_component_selector``: the same function
    that wrote the field is the one that reads it, so a stored discriminator can
    never name a row the resolver would pick differently. Testing subtree
    containment instead would be looser and would also match the row's
    container, which holds every row's text.
    """
    if not value:
        return False
    return discriminative_descendant_text(elem, elements) == value


def _matches_nearby(elem: UIElement, value: str, by_id: dict[str, UIElement]) -> bool:
    if not value:
        return True
    # Find an android:id/title descendant with matching normalized text.
    stack = list(elem.children)
    seen: set[str] = set()
    while stack:
        cid = stack.pop()
        if cid in seen:
            continue
        seen.add(cid)
        child = by_id.get(cid)
        if child is None:
            continue
        if child.resource_id == "android:id/title" and _normalize_text(child.text) == value:
            return True
        stack.extend(child.children)
    return False


def _matches_ancestor_chain(elem: UIElement, chain: list[str], by_id: dict[str, UIElement]) -> bool:
    if not chain:
        return True
    expected = list(chain)
    cur_id = elem.parent_id
    walked: list[str] = []
    while cur_id and len(walked) < len(expected):
        parent = by_id.get(cur_id)
        if parent is None:
            break
        rid_tail = (parent.resource_id or "").split(":id/")[-1] if parent.resource_id else ""
        token = f"{_class_leaf(parent.class_name)}#{rid_tail}"
        walked.append(token)
        cur_id = parent.parent_id
    walked.reverse()
    # ancestor_chain in selector is top-down; require it to be a suffix of walked.
    if len(walked) < len(expected):
        return False
    return walked[-len(expected) :] == expected


def resolve_selector(
    selector: dict | None,
    elements: list[UIElement],
    *,
    action_type: ActionType | None = None,
) -> ResolutionResult:
    """Resolve ``selector`` against the live ``elements`` list.

    Returns a :class:`ResolutionResult` with status ∈ {MATCH, MISSING,
    AMBIGUOUS}. Callers must treat any non-MATCH status as a refusal to
    execute — the explorer never clicks a guessed candidate.

    When ``action_type`` is provided, candidates are pre-filtered to
    those that are (a) enabled, (b) visible (non-degenerate bounds), and
    (c) action-compatible with the requested action type (e.g. CLICK
    requires ``is_clickable``). This avoids picking a disabled or
    off-screen element that happens to match the rid.
    """
    if not selector or not elements:
        return ResolutionResult(ResolutionStatus.MISSING, None, 0, "empty selector or screen")
    by_id = {e.element_id: e for e in elements}
    pool = [e for e in elements if _is_candidate(e, action_type)]
    if not pool:
        return ResolutionResult(
            ResolutionStatus.MISSING,
            None,
            0,
            "no enabled / visible / action-compatible candidates",
        )

    rid = selector.get("resource_id") or ""
    text = _normalize_text(selector.get("text"))
    cd = selector.get("content_description") or ""
    nearby = _normalize_text(selector.get("nearby_text"))
    row_disc = selector.get("row_discriminator") or ""
    cls = selector.get("class_name") or ""
    chain_raw = selector.get("ancestor_chain") or []
    chain = list(chain_raw) if isinstance(chain_raw, list) else []

    # Seed candidate pool with elements matching the strongest field present.
    candidates: list[UIElement]
    seed_field: str
    if rid:
        candidates = [e for e in pool if _matches_field(e, "resource_id", rid)]
        seed_field = "resource_id"
    elif text:
        candidates = [e for e in pool if _matches_text_strong(e, text, by_id)]
        seed_field = "text"
    elif cd:
        candidates = [e for e in pool if _matches_field(e, "content_description", cd)]
        seed_field = "content_description"
    elif nearby:
        candidates = [e for e in pool if _matches_nearby(e, nearby, by_id)]
        seed_field = "nearby_text"
    elif row_disc:
        candidates = [e for e in pool if _matches_row_discriminator(e, row_disc, elements)]
        seed_field = "row_discriminator"
    elif cls:
        candidates = [e for e in pool if _matches_field(e, "class_name", cls)]
        seed_field = "class_name"
    else:
        return ResolutionResult(
            ResolutionStatus.MISSING,
            None,
            0,
            "selector has no usable identity field",
        )

    seed_count = len(candidates)
    if seed_count == 0:
        return ResolutionResult(
            ResolutionStatus.MISSING, None, 0, f"no element matched {seed_field}"
        )
    if seed_count == 1:
        return ResolutionResult(ResolutionStatus.MATCH, candidates[0], 1)

    # Disambiguate: apply remaining specified fields as filters in priority order.
    # Each filter must STRICTLY shrink the pool to one element to claim MATCH.
    filters: list[tuple[str, str]] = []
    if seed_field != "text" and text:
        filters.append(("text", text))
    if seed_field != "content_description" and cd:
        filters.append(("content_description", cd))
    if seed_field != "nearby_text" and nearby:
        filters.append(("nearby_text", nearby))
    if seed_field != "row_discriminator" and row_disc:
        filters.append(("row_discriminator", row_disc))
    if seed_field != "class_name" and cls:
        filters.append(("class_name", cls))

    filtered = candidates
    for key, value in filters:
        if key == "nearby_text":
            filtered = [e for e in filtered if _matches_nearby(e, value, by_id)]
        elif key == "row_discriminator":
            filtered = [e for e in filtered if _matches_row_discriminator(e, value, elements)]
        elif key == "text":
            filtered = [e for e in filtered if _matches_text_strong(e, value, by_id)]
        else:
            filtered = [e for e in filtered if _matches_field(e, key, value)]
        if len(filtered) <= 1:
            break

    if chain and len(filtered) > 1:
        filtered = [e for e in filtered if _matches_ancestor_chain(e, chain, by_id)]

    if len(filtered) == 1:
        return ResolutionResult(ResolutionStatus.MATCH, filtered[0], seed_count)
    if len(filtered) == 0:
        return ResolutionResult(
            ResolutionStatus.MISSING,
            None,
            seed_count,
            f"{seed_count} {seed_field} candidates, disambiguation eliminated all",
        )
    return ResolutionResult(
        ResolutionStatus.AMBIGUOUS,
        None,
        len(filtered),
        f"{len(filtered)} candidates share {seed_field} after disambiguation",
    )


def _matches_text_strong(elem: UIElement, text: str, by_id: dict[str, UIElement]) -> bool:
    """Own text equality OR Preference-row title-descendant equality."""
    if _normalize_text(elem.text) == text:
        return True
    if not elem.text:
        return _matches_nearby(elem, text, by_id)
    return False

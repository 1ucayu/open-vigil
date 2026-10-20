"""Agent-agnostic action binding: proposed action -> canonical bound action.

An agent proposes an action however it likes (an element handle, a selector,
free-form identity fields). Before that action can be verified it must be bound
to Vigil's *existing* canonical action identity so that
:meth:`AppFSM.resolve_transition` / :func:`canonical_action_key` can resolve it.

Binding rules (paper-aligned):

- Runtime XML evidence is the source of truth. Binding always confirms a
  concrete live ``UIElement`` on the current screen and *synthesizes*
  authoritative stable fields from it via ``build_component_selector``. It never
  trusts caller-supplied identity without live-screen confirmation and never
  fabricates target fields.
- Global navigation (``navigate_back`` / ``navigate_home``) needs no target.
- Anything that cannot be confirmed to a stable identity on the live screen is
  ``UNBOUND`` (the monitor returns UNCERTAIN, never a silent ALLOW).

This module does not change ``canonical_action_key`` and introduces no new
action alphabet; it only maps a proposal into the fields that identity already
consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from vigil.core.ui_selectors import (
    _class_leaf,
    build_component_selector,
    selector_has_stable_identity,
)
from vigil.models.action import ActionType
from vigil.models.fsm import is_value_payload_action
from vigil.models.state import RawScreen, UIElement
from vigil.neuro.selector_resolution import (
    ResolutionStatus,
    _is_candidate,
    resolve_selector,
)

# Global actions that resolve without any element target (see fsm._GLOBAL_ACTION_TYPES).
_GLOBAL_ACTION_TYPES = {ActionType.NAVIGATE_BACK.value, ActionType.NAVIGATE_HOME.value}


def _is_bounds(value: Any) -> bool:
    """True for a well-formed [left, top, right, bottom] rectangle."""
    return (
        isinstance(value, list | tuple)
        and len(value) == 4
        and all(isinstance(v, int) for v in value)
    )


class BindingStatus(StrEnum):
    """Outcome of binding a proposed action to a live screen element."""

    MATCH = "match"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    UNBOUND = "unbound"


class ProposedAction(BaseModel):
    """An agent-proposed action prior to binding.

    Extra keys are preserved (``extra="allow"``) so nothing an agent supplies is
    dropped. ``type`` is the ``ActionType`` string value. Stable identity fields
    are all optional — the binder confirms whatever is present against the live
    screen.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    target: str | None = None
    resource_id: str | None = None
    target_resource_id: str | None = None
    target_text: str | None = None
    target_content_desc: str | None = None
    target_class_name: str | None = None
    target_selector: dict[str, Any] | None = None
    text: str | None = None
    value: str | None = None
    input_text: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProposedAction:
        """Build from a plain action dict, accepting ``action_type`` as ``type``."""
        payload = dict(data)
        if "type" not in payload and "action_type" in payload:
            payload["type"] = payload["action_type"]
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """Emit a plain action dict, dropping None fields (keeps extras)."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


@dataclass(frozen=True)
class ActionBindingResult:
    """Result of :meth:`ActionBinder.bind`.

    Attributes:
        status: MATCH / MISSING / AMBIGUOUS / UNBOUND.
        bound_action: The action dict to feed to ``resolve_transition``. On a
            non-MATCH status it is the normalized input (never fabricated
            identity), useful for diagnostics.
        reason: Human-readable explanation for diagnostics / trace metadata.
    """

    status: BindingStatus
    bound_action: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class ActionBinder:
    """Binds proposed actions into Vigil's canonical action identity."""

    def bind(
        self, proposed_action: dict[str, Any] | ProposedAction, screen: RawScreen
    ) -> ActionBindingResult:
        """Bind ``proposed_action`` to a live element on ``screen``."""
        action = self._normalize(proposed_action)
        atype = action.get("type")

        # Rule 1: global navigation needs no target binding.
        if atype in _GLOBAL_ACTION_TYPES:
            return ActionBindingResult(
                BindingStatus.MATCH, action, "global action; no target binding required"
            )

        action_type = self._as_action_type(atype)

        # Rule 2: an explicit live element_id handle — confirm it exists, enrich.
        target = action.get("target")
        by_id = {e.element_id: e for e in screen.elements}
        if isinstance(target, str) and target in by_id:
            return self._bind_to_element(action, by_id[target], screen, "live element_id")

        # Rule 3: a stable-identity target_selector — resolve against the screen.
        selector = action.get("target_selector")
        if isinstance(selector, dict) and selector_has_stable_identity(selector):
            if is_value_payload_action(action):
                selector = dict(selector)
                selector.pop("text", None)
            if selector_has_stable_identity(selector):
                return self._resolve_and_bind(
                    action, selector, screen, action_type, "target_selector"
                )

        # Rule 4: other caller identity fields — synthesize a selector, resolve.
        synthesized = self._selector_from_fields(action)
        if selector_has_stable_identity(synthesized):
            return self._resolve_and_bind(
                action, synthesized, screen, action_type, "stable identity fields"
            )

        # Rule 5: an identity-less repeated row — locate it geometrically, then let
        # the live element supply the identity the proposal could not.
        #
        # The agent's element snapshot is flat: for a row carrying no resource_id,
        # text, or content-description, every row on the screen produces a byte
        # identical snapshot, so the proposal cannot say WHICH row it meant.
        # ``target_bounds`` is the only channel that can, and it is resolution-only
        # (never in ``fsm._ACTION_KEY_FIELDS``, which is a closed allowlist).
        bounds = action.get("target_bounds")
        if _is_bounds(bounds):
            return self._bind_by_bounds(action, list(bounds), screen, action_type)

        # Rule 6: nothing confirmable on the live screen.
        return ActionBindingResult(
            BindingStatus.UNBOUND,
            action,
            "no confirmable stable identity on the live screen",
        )

    def _bind_by_bounds(
        self,
        action: dict[str, Any],
        bounds: list[int],
        screen: RawScreen,
        action_type: ActionType | None,
    ) -> ActionBindingResult:
        """Bind to the one live element occupying exactly ``bounds``.

        Exact equality only. A mismatch is UNBOUND, never the nearest element:
        the agent named a rectangle, and if nothing sits at that rectangle the
        screen has moved under the proposal. Guessing the closest row is exactly
        the wrong-item error the shield exists to prevent.
        """
        cls = action.get("target_class_name") or ""
        hits = [
            e
            for e in screen.elements
            if list(e.bounds or []) == bounds and _is_candidate(e, action_type)
        ]
        if cls and len(hits) > 1:
            narrowed = [
                e
                for e in hits
                if e.class_name == cls or _class_leaf(e.class_name) == _class_leaf(cls)
            ]
            if narrowed:
                hits = narrowed
        if len(hits) == 1:
            return self._bind_to_element(action, hits[0], screen, "target bounds")
        if not hits:
            return ActionBindingResult(
                BindingStatus.UNBOUND,
                action,
                f"no live element occupies the proposed bounds {bounds}",
            )
        return ActionBindingResult(
            BindingStatus.AMBIGUOUS,
            action,
            f"{len(hits)} live elements share the proposed bounds {bounds}",
        )

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _normalize(proposed_action: dict[str, Any] | ProposedAction) -> dict[str, Any]:
        """Coerce to a plain dict, normalize ``type`` and ``input_text``."""
        if isinstance(proposed_action, ProposedAction):
            action = proposed_action.to_dict()
        else:
            action = dict(proposed_action)
        if "type" not in action and "action_type" in action:
            action["type"] = action["action_type"]
        # Normalize input_text -> text so the canonical action key is stable.
        if action.get("input_text") and not action.get("text"):
            action["text"] = action["input_text"]
        return action

    @staticmethod
    def _as_action_type(atype: Any) -> ActionType | None:
        """Map a type string to ActionType for the resolver's capability filter."""
        if not isinstance(atype, str):
            return None
        try:
            return ActionType(atype)
        except ValueError:
            return None

    @staticmethod
    def _selector_from_fields(action: dict[str, Any]) -> dict[str, Any]:
        """Build a selector dict from caller-supplied top-level identity fields."""
        target_text = "" if is_value_payload_action(action) else action.get("target_text") or ""
        return {
            "resource_id": action.get("target_resource_id") or action.get("resource_id") or "",
            "text": target_text,
            "content_description": action.get("target_content_desc") or "",
            "class_name": action.get("target_class_name") or "",
            "nearby_text": "",
            "row_discriminator": "",
            "ancestor_chain": [],
        }

    def _resolve_and_bind(
        self,
        action: dict[str, Any],
        selector: dict[str, Any],
        screen: RawScreen,
        action_type: ActionType | None,
        label: str,
    ) -> ActionBindingResult:
        result = resolve_selector(selector, screen.elements, action_type=action_type)
        if result.status is ResolutionStatus.MATCH and result.element is not None:
            return self._bind_to_element(action, result.element, screen, label)
        if result.status is ResolutionStatus.AMBIGUOUS:
            return ActionBindingResult(
                BindingStatus.AMBIGUOUS, action, f"{label} ambiguous: {result.reason}"
            )
        return ActionBindingResult(
            BindingStatus.MISSING, action, f"{label} missing: {result.reason}"
        )

    @staticmethod
    def _bind_to_element(
        action: dict[str, Any], element: UIElement, screen: RawScreen, label: str
    ) -> ActionBindingResult:
        """Synthesize authoritative stable identity from a confirmed live element.

        Mirrors ``Action.to_fsm_dict`` semantics: only non-empty stable fields are
        written, so the canonical action key stays compatible with FSM-stored
        transitions. ``target`` is set to the live element_id.
        """
        selector = build_component_selector(element, screen.elements)
        bound = dict(action)
        bound["target"] = element.element_id
        value_payload_action = is_value_payload_action(bound)
        if value_payload_action:
            selector = dict(selector)
            selector.pop("text", None)

        rid = selector.get("resource_id") or ""
        if rid:
            bound["resource_id"] = rid
            bound["target_resource_id"] = rid
        else:
            bound.pop("resource_id", None)
            bound.pop("target_resource_id", None)

        if selector.get("text") and not value_payload_action:
            bound["target_text"] = selector["text"]
        else:
            bound.pop("target_text", None)

        if selector.get("content_description"):
            bound["target_content_desc"] = selector["content_description"]
        else:
            bound.pop("target_content_desc", None)

        if selector.get("class_name"):
            bound["target_class_name"] = selector["class_name"]
        else:
            bound.pop("target_class_name", None)

        bound["target_selector"] = selector
        return ActionBindingResult(BindingStatus.MATCH, bound, f"bound via {label}")

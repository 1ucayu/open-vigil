"""Per-state invariant evidence view for contract-first invariant generation.

An :class:`InvariantEvidence` is the deterministic, LLM-free evidence bundle for one
*arrival* FSM state that the invariant/guard candidate generator consumes. It mirrors
:mod:`vigil.neuro.guard_evidence` (which is per-transition) but is keyed per-state. It
joins:

- the target-state identity / page-function annotations,
- the arrival-state :class:`~vigil.neuro.guard_registry.WidgetRegistry`,
- every resolvable raw-screen observation of the state (repeated visits are stronger
  evidence than a single screenshot, and are what the admission layer replays against),
- incoming / outgoing transition summaries used only to avoid predecessor-specific or
  action-affordance candidates,
- static-prior hints (``priors only, never proof``).

This module only *reads* the FSM and raw trace screens. It adds no edges, changes no
state identity / transition evidence, compiles no DSL, validates no admission, and calls no
LLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from vigil.neuro.guard_evidence import exploration_payload_values, static_prior_hints
from vigil.neuro.guard_registry import WidgetRegistry, _as_dict, build_widget_registry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vigil.models.fsm import AbstractState, AppFSM, Transition
    from vigil.neuro.app_prior import AppPrior


class TransitionSummary(BaseModel):
    """A lightweight incoming/outgoing transition view for one state."""

    source_state_id: str
    target_state_id: str
    action: dict[str, Any] = Field(default_factory=dict)


class InvariantEvidence(BaseModel):
    """Deterministic per-state evidence bundle for invariant/guard synthesis."""

    target_state_id: str
    target_state_name: str = ""
    window_name: str = ""
    container_type: str = ""
    page_function: str = ""
    display_name: str = ""
    visual_alt_text: str = ""
    raw_screen_ids: list[str] = Field(default_factory=list)
    observation_count: int = 0
    observations: list[dict[str, Any]] = Field(default_factory=list)
    arrival_registry: WidgetRegistry
    incoming: list[TransitionSummary] = Field(default_factory=list)
    outgoing: list[TransitionSummary] = Field(default_factory=list)
    static_prior_hints: list[str] = Field(default_factory=list)
    exploration_payload_values: list[str] = Field(default_factory=list)


def _resolve_observations(
    state: AbstractState,
    raw_screens: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return every raw-screen dict of ``state`` that resolves in ``raw_screens``."""
    observations: list[dict[str, Any]] = []
    for sid in dict.fromkeys(state.evidence.raw_screen_ids):
        candidate = raw_screens.get(sid)
        if candidate is None:
            continue
        observation = dict(_as_dict(candidate))
        observation["screen_id"] = observation.get("screen_id") or sid
        observations.append(observation)
    return observations


def _transition_summary(transition: Transition) -> TransitionSummary:
    return TransitionSummary(
        source_state_id=transition.source,
        target_state_id=transition.target,
        action=transition.action,
    )


def build_invariant_evidence(
    fsm: AppFSM,
    state: AbstractState,
    raw_screens: dict[str, Any],
    app_prior: AppPrior | None = None,
) -> InvariantEvidence:
    """Build the :class:`InvariantEvidence` view for a single arrival state."""
    registry = build_widget_registry(state, raw_screens, app_prior)
    observations = _resolve_observations(state, raw_screens)

    incoming: list[TransitionSummary] = []
    outgoing: list[TransitionSummary] = []
    probe_values: list[str] = []
    for transition in fsm.transitions:
        if transition.target == state.state_id:
            incoming.append(_transition_summary(transition))
            for value in exploration_payload_values(transition):
                if value not in probe_values:
                    probe_values.append(value)
        if transition.source == state.state_id:
            outgoing.append(_transition_summary(transition))

    return InvariantEvidence(
        target_state_id=state.state_id,
        target_state_name=state.name,
        window_name=state.android_context.window_type or "",
        container_type=state.abstraction.container_type.value,
        page_function=state.annotations.page_function,
        display_name=state.annotations.display_name,
        visual_alt_text=state.annotations.alt_text,
        raw_screen_ids=list(state.evidence.raw_screen_ids),
        observation_count=len(observations),
        observations=observations,
        arrival_registry=registry,
        incoming=incoming,
        outgoing=outgoing,
        static_prior_hints=static_prior_hints(state, app_prior, registry),
        exploration_payload_values=probe_values,
    )


def build_all_invariant_evidence(
    fsm: AppFSM,
    raw_screens: dict[str, Any],
    app_prior: AppPrior | None = None,
) -> list[InvariantEvidence]:
    """Build :class:`InvariantEvidence` for every state in ``fsm`` (insertion order)."""
    return [
        build_invariant_evidence(fsm, state, raw_screens, app_prior)
        for state in fsm.states.values()
    ]

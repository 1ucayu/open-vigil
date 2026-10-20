"""Apply paper-defined EFSM operations to one observed runtime transition.

This module owns the core admission sequence for a real ``<pre, action, post>``
observation, decoupled from any agent, benchmark, or transport concern. Given an
:class:`~vigil.models.fsm.AppFSM`, the real ``<pre, action, post>`` observation (as
:class:`~vigil.models.state.RawScreen` objects and a canonical bound action), and a
hydrated raw-screen evidence store, it drives the existing
:class:`~vigil.neuro.evolution.RuntimeEvolutionEngine`:

    record_fallback_step
        -> admit_state(pre) -> admit_state(post)
        -> admit_transition  (admit | corroborate | abstraction-refine | reject)

and returns *factual* admission results. It never converts M3A actions, correlates
proposals/receipts, formats JSON, or performs semantic (LLM) promotion — those are
integration concerns. New edges carry runtime-fallback origin provenance and a PENDING
placeholder guard contract: they have not been through guard admission, so the
verifier answers UNCERTAIN for them (present-but-pending contract) until an admitted
contract discharges the placeholder.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from vigil.models.evolution import EvolutionOperation, EvolutionOutcome
from vigil.models.fsm import AppFSM, is_runtime_fallback_origin
from vigil.models.state import RawScreen
from vigil.neuro.evolution import AdmissionResult, RuntimeEvolutionEngine


class ObservedTransitionResult(BaseModel):
    """Factual result of applying an observed transition to the EFSM."""

    model_changed: bool
    operation: EvolutionOperation | None
    outcome: EvolutionOutcome
    source_admission: AdmissionResult
    target_admission: AdmissionResult
    transition_admission: AdmissionResult
    pre_snapshot: RawScreen
    post_snapshot: RawScreen
    provisional: bool
    efsm_revision_before: int
    efsm_revision_after: int
    rejection_reason: str = ""


def apply_observed_transition(
    fsm: AppFSM,
    pre_screen: RawScreen,
    bound_action: dict[str, Any],
    post_screen: RawScreen,
    raw_screens: dict[str, RawScreen],
) -> ObservedTransitionResult:
    """Record one executed observation and apply exactly one canonical operation.

    ``bound_action`` must already be the canonical action bound to ``pre_screen``; the
    engine re-validates the binding internally, so a stale/ambiguous action is rejected
    rather than trusted. ``raw_screens`` is the hydrated evidence store (exploration trace
    + persisted runtime captures) so existing states' provenance ids resolve for any
    abstraction refinement.

    The returned ``pre_snapshot``/``post_snapshot`` are the engine-assigned evidence
    captures the caller should persist so their ids stay resolvable in later calls.
    """
    engine = RuntimeEvolutionEngine(fsm, raw_screens)
    revision_before = fsm.efsm_revision

    step = engine.record_fallback_step(pre_screen, bound_action, post_screen)
    pre_snapshot = step.pre_screen
    post_snapshot = step.post_screen

    # A step is only admissible when both endpoints classified in-app (EXISTING/NOVEL);
    # otherwise the engine never enrolled it as pending evidence. Guard on that so a
    # rejected step (e.g. an external post-screen) never invents an orphan state.
    step_is_pending = any(recorded is step for recorded in engine.pending_steps())
    if not step_is_pending:
        return _rejected_result(
            pre_snapshot,
            post_snapshot,
            revision_before,
            fsm.efsm_revision,
            reason="fallback step is not admissible evidence (endpoint not in-app)",
        )

    source_admission = engine.admit_state(pre_snapshot)
    target_admission = engine.admit_state(post_snapshot)
    transition_admission = engine.admit_transition(step)

    operation = _classify_operation(source_admission, target_admission, transition_admission)
    model_changed = operation in {
        EvolutionOperation.COVERAGE_EXPANSION,
        EvolutionOperation.ABSTRACTION_REFINEMENT,
    }
    outcome = (
        EvolutionOutcome.APPLIED
        if model_changed
        else EvolutionOutcome.CORROBORATED
        if operation is EvolutionOperation.EVIDENCE_ENRICHMENT
        else EvolutionOutcome.REJECTED
    )
    rejection_reason = "" if transition_admission.admitted else transition_admission.reason

    return ObservedTransitionResult(
        model_changed=model_changed,
        operation=operation,
        outcome=outcome,
        source_admission=source_admission,
        target_admission=target_admission,
        transition_admission=transition_admission,
        pre_snapshot=pre_snapshot,
        post_snapshot=post_snapshot,
        # Provisional iff an edge was admitted and it originates from runtime evolution
        # rather than offline construction. Read from origin provenance.
        provisional=_edge_provisional(transition_admission),
        efsm_revision_before=revision_before,
        efsm_revision_after=fsm.efsm_revision,
        rejection_reason=rejection_reason,
    )


def _classify_operation(
    source_admission: AdmissionResult,
    target_admission: AdmissionResult,
    transition_admission: AdmissionResult,
) -> EvolutionOperation | None:
    if transition_admission.admitted:
        if transition_admission.created:
            if transition_admission.refined:
                return EvolutionOperation.ABSTRACTION_REFINEMENT
            return EvolutionOperation.COVERAGE_EXPANSION
        return EvolutionOperation.EVIDENCE_ENRICHMENT
    # Transition not admitted: a coverage expansion still occurred if a state was created.
    if source_admission.created or target_admission.created:
        return EvolutionOperation.COVERAGE_EXPANSION
    return None


def _edge_provisional(transition_admission: AdmissionResult) -> bool:
    """True iff an admitted edge was created by runtime evolution, not offline construction.

    "Provisional" is a structural fact read from the edge's *origin* provenance
    (:func:`~vigil.models.fsm.is_runtime_fallback_origin`): the edge exists only because a
    runtime fallback observed it. Such an edge is created with a PENDING placeholder
    contract, so the verifier answers UNCERTAIN for it until admission discharges the
    placeholder; the provenance itself is reporting, never a gate, and is deliberately
    not consulted by any verdict path.

    A newly admitted runtime edge is provisional. An offline edge that this observation
    merely corroborated is not — corroboration adds evidence but does not change where the
    edge came from. No admitted edge (rejected or state-only) is not provisional.
    """
    edge = transition_admission.transition
    return bool(
        transition_admission.admitted and edge is not None and is_runtime_fallback_origin(edge)
    )


def _rejected_result(
    pre_snapshot: RawScreen,
    post_snapshot: RawScreen,
    revision_before: int,
    revision_after: int,
    *,
    reason: str,
) -> ObservedTransitionResult:
    empty = AdmissionResult(admitted=False, reason=reason)
    return ObservedTransitionResult(
        model_changed=False,
        operation=None,
        outcome=EvolutionOutcome.REJECTED,
        source_admission=empty,
        target_admission=empty,
        transition_admission=empty,
        pre_snapshot=pre_snapshot,
        post_snapshot=post_snapshot,
        provisional=False,
        efsm_revision_before=revision_before,
        efsm_revision_after=revision_after,
        rejection_reason=reason,
    )

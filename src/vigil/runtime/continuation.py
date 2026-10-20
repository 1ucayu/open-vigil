"""Typed continuation boundaries for verifier coverage gaps.

``SemanticGapContinuationClassifier`` is the non-executing boundary for a known,
structurally trusted transition whose offline semantic guard asset is incomplete.  It
emits an eligibility record for the outer agent/evolution policy; it never executes the
action, calls an LLM, or mutates the EFSM.

When runtime localization fails, or a runtime-evolved state has incomplete outgoing
coverage, the verifier returns ``UNCERTAIN``. That is not a terminal task failure —
``TopologyContinuationClassifier`` is the smallest Vigil-owned callable boundary that
decides what to do next:

- an action whose safety follows from its **type alone** (``navigate_back``,
  ``navigate_home``, ``scroll_up``, ``scroll_down``) may be authorized as a recovery /
  risk-bounded fallback;
- any component action (``click`` / ``long_press`` / ``input_text`` / unclassified)
  requires user confirmation and is never auto-authorized.

Text/resource/label lexicons are deliberately absent: only the action type authorizes
automatic continuation.

``classify_post_action_witness`` is the third boundary and the only one that looks at an
observation the agent has *already* executed: after the evidence-only post-action check
has run, it decides whether the landed screen is an admissible evolution witness (a
successor mismatch, a novel successor, or a concretely refuted invariant). It is pure
classification — it never mutates the EFSM, executes anything, or calls an LLM; the
caller submits an accepted witness to the existing observed-transition service.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from vigil.models.fsm import (
    RUNTIME_EVOLUTION_SOURCE,
    AppFSM,
    TransitionLookupStatus,
)
from vigil.models.state import RawScreen
from vigil.neuro.behavioral_signature import refinement_identity_hash
from vigil.neuro.evolution import ObservationClass, RuntimeEvolutionEngine
from vigil.runtime.action_binder import ActionBinder, BindingStatus, ProposedAction
from vigil.symbolic.decision_engine import DecisionEngine
from vigil.symbolic.fsm_checker import (
    AUTO_ELIGIBLE_ACTION_TYPES,
    SemanticGuardCoverageGap,
    VerificationOutput,
    VerifyReason,
    is_conservative_refusal,
)
from vigil.symbolic.state_locator import raw_screen_to_quotient_input

# Safety of these actions follows from their type: they are reversible / non
# side-effecting GUI-control actions. Everything else (click / long_press /
# input_text / unclassified) requires user confirmation on an unknown screen.
# Defined in ``symbolic.fsm_checker`` (which this module imports) and re-exported
# here. Imported, not redefined: there is exactly one set. An UNCERTAIN names a
# coverage gap, never a proof of safety, so this action-type gate is what stands
# between a missing edge and a side effect.


class ContinuationRefusal(StrEnum):
    """Machine-readable reason an evolution witness was not admitted.

    One compact vocabulary for both continuation boundaries: the pre-action topology
    continuation (which authorizes at most one unverified evidence execution) and the
    post-action witness path (which submits an already-executed observation to the
    evolution engine). Every value names a *refusal*, never an outcome of an engine
    call — the engine reports its own outcome through
    :class:`~vigil.models.evolution.EvolutionOutcome`.

    These strings are recorded in agent-run rows, so they are API: extend the enum
    rather than re-wording an existing value.
    """

    # -- shared source/observation refusals ---------------------------------
    OBSERVATION_AMBIGUOUS = "observation_ambiguous"
    OBSERVATION_EXTERNAL = "observation_external"
    OBSERVATION_UNCLASSIFIABLE = "observation_unclassifiable"
    ACTION_UNBOUND = "action_unbound"
    POLICY_GATED = "policy_gated"

    # -- pre-action topology continuation -----------------------------------
    NOT_TOPOLOGY_GAP = "not_topology_gap"
    ACTION_ALREADY_MODELED = "action_already_modeled"
    SOURCE_NOT_A_GAP = "source_not_a_gap"
    SOURCE_VERDICT_RACE = "source_verdict_race"
    CONFIRMATION_REQUIRED = "confirmation_required"
    ACTION_NOT_AUTO_ELIGIBLE = "action_not_auto_eligible"
    FOREGROUND_CONTEXT_MISSING = "foreground_context_missing"

    # -- post-action witness admission --------------------------------------
    NO_EXECUTION_CONTEXT = "no_execution_context"
    EXECUTION_CONTEXT_MISMATCH = "execution_context_mismatch"
    NO_WITNESS = "no_witness"
    POST_LOCALIZATION_INCONCLUSIVE = "post_localization_inconclusive"
    INVARIANT_INCONCLUSIVE = "invariant_inconclusive"
    WITNESS_CLASS_ALREADY_COVERED = "witness_class_already_covered"
    WITNESS_ATTRIBUTION_UNAVAILABLE = "witness_attribution_unavailable"
    WITNESS_ADMISSION_ERROR = "witness_admission_error"


def is_auto_eligible(action: dict[str, Any]) -> bool:
    """True iff the action type alone authorizes automatic continuation."""
    return (action.get("type") or action.get("action_type")) in AUTO_ELIGIBLE_ACTION_TYPES


def classify_continuation_source(
    fsm: AppFSM,
    engine: RuntimeEvolutionEngine,
    screen: RawScreen,
    action: dict[str, Any],
) -> tuple[bool, bool, str, ContinuationRefusal | None]:
    """Single source-eligibility policy for continuation boundaries.

    Returns ``(eligible, requires_confirmation, reason, refusal_code)``. The code is
    ``None`` for an eligible source and otherwise the machine-readable form of
    ``reason``. Eligibility (is this a legal evolution source?) is decoupled from
    authorization (has this execution been confirmed?):

    - ``NOVEL`` runtime observation -> ``(True, False)``;
    - runtime-evolved ``EXISTING`` state with a missing outgoing edge -> ``(True, False)``;
    - offline ``EXISTING`` state with a missing edge -> ``(True, False)``. Missing
      transition evidence never refutes, so the refusal here is conservative rather than
      proof-backed and there is no decisive refusal left for the action to
      self-authorize around. What still bounds the risk is the action-type gate in
      :func:`is_auto_eligible`, not the verdict;
    - everything else (EXTERNAL/AMBIGUOUS/UNCLASSIFIABLE, or an action the state already
      models/ambiguously resolves) -> ineligible.
    """
    classification = engine.classify_observation(screen)
    if classification.kind is ObservationClass.NOVEL:
        return True, False, "novel runtime observation", None
    if classification.kind is not ObservationClass.EXISTING or not classification.state_id:
        return (
            False,
            False,
            f"observation is {classification.kind.value}, not admissible app state",
            _OBSERVATION_REFUSALS.get(
                classification.kind, ContinuationRefusal.OBSERVATION_UNCLASSIFIABLE
            ),
        )

    state = fsm.states[classification.state_id]
    lookup = fsm.resolve_transition(classification.state_id, action)
    if state.evidence.construction_source == RUNTIME_EVOLUTION_SOURCE:
        if lookup.status is TransitionLookupStatus.NO_MATCH:
            return True, False, "runtime-evolved state has incomplete outgoing coverage", None
        return (
            False,
            False,
            "runtime-evolved state already models or ambiguously resolves this action",
            ContinuationRefusal.ACTION_ALREADY_MODELED,
        )
    if lookup.status is TransitionLookupStatus.NO_MATCH:
        return True, False, "coverage expansion on an existing offline state", None
    return (
        False,
        False,
        "screen localizes to an offline state; no fallback needed",
        ContinuationRefusal.SOURCE_NOT_A_GAP,
    )


# Observation classes that are never an admissible evolution source.
_OBSERVATION_REFUSALS: dict[ObservationClass, ContinuationRefusal] = {
    ObservationClass.AMBIGUOUS: ContinuationRefusal.OBSERVATION_AMBIGUOUS,
    ObservationClass.EXTERNAL: ContinuationRefusal.OBSERVATION_EXTERNAL,
    ObservationClass.UNCLASSIFIABLE: ContinuationRefusal.OBSERVATION_UNCLASSIFIABLE,
}


class ContinuationEligibility(StrEnum):
    """Whether an outer policy may treat this gap as an evolution candidate."""

    ELIGIBLE_FOR_EVOLUTION = "eligible_for_evolution"
    INELIGIBLE = "ineligible"


class SemanticGapContinuationOutcome(BaseModel):
    """Pure classification of a semantic-gap action proposal."""

    eligibility: ContinuationEligibility
    reason: VerifyReason | None = None
    details: str = ""
    semantic_guard_gap: SemanticGuardCoverageGap | None = None
    current_state_id: str | None = None
    target_state_id: str | None = None
    bound_action: dict[str, Any] | None = None


class SemanticGapContinuationClassifier:
    """Classify structurally trusted semantic coverage gaps without executing them."""

    def __init__(self, fsm: AppFSM) -> None:
        self._fsm = fsm
        self._binder = ActionBinder()

    def classify(
        self,
        verification: VerificationOutput,
        screen: RawScreen,
        proposed_action: dict[str, Any] | ProposedAction,
    ) -> SemanticGapContinuationOutcome:
        """Return eligibility only for a known, uniquely bound, trusted gap."""
        if (
            not is_conservative_refusal(verification)
            or verification.reason is not VerifyReason.SEMANTIC_GUARD_COVERAGE_GAP
            or verification.semantic_guard_gap is None
        ):
            return self._ineligible(
                verification,
                "verification is not a typed semantic guard coverage gap",
            )

        state_id = verification.current_state_id
        if state_id is None:
            return self._ineligible(
                verification,
                "verification does not carry a deterministically known state",
            )

        binding = self._binder.bind(proposed_action, screen)
        if binding.status is not BindingStatus.MATCH:
            return self._ineligible(
                verification,
                f"action binding is {binding.status.value}, not match",
            )

        lookup = self._fsm.resolve_transition(state_id, binding.bound_action)
        if lookup.status is not TransitionLookupStatus.MATCH or lookup.transition is None:
            return self._ineligible(
                verification,
                f"bound action transition resolution is {lookup.status.value}, not match",
            )
        transition = lookup.transition

        return SemanticGapContinuationOutcome(
            eligibility=ContinuationEligibility.ELIGIBLE_FOR_EVOLUTION,
            reason=VerifyReason.SEMANTIC_GUARD_COVERAGE_GAP,
            details=("known state and uniquely bound transition"),
            semantic_guard_gap=verification.semantic_guard_gap,
            current_state_id=state_id,
            target_state_id=transition.target,
            bound_action=binding.bound_action,
        )

    @staticmethod
    def _ineligible(
        verification: VerificationOutput,
        details: str,
        *,
        bound_action: dict[str, Any] | None = None,
    ) -> SemanticGapContinuationOutcome:
        return SemanticGapContinuationOutcome(
            eligibility=ContinuationEligibility.INELIGIBLE,
            reason=verification.reason,
            details=details,
            semantic_guard_gap=verification.semantic_guard_gap,
            current_state_id=verification.current_state_id,
            target_state_id=verification.target_state_id,
            bound_action=bound_action,
        )


class TopologyAuthorization(StrEnum):
    """Outer-policy authorization for a topology (state/transition) coverage gap."""

    AUTHORIZED_AUTO = "authorized_auto"  # action-type-safe on a topology gap; one evidence run
    CONFIRMATION_REQUIRED = "confirmation_required"  # component action / offline gap; stays blocked
    REJECTED = "rejected"  # not a topology coverage gap, or unbindable/inadmissible


class TopologySourceKind(StrEnum):
    """Which topology coverage gap the verifier + observation agree on."""

    NOVEL_UNLOCALIZED = "novel_unlocalized"  # UNCERTAIN + STATE_UNKNOWN
    RUNTIME_EVOLVED_INCOMPLETE_COVERAGE = (
        "runtime_evolved_incomplete_coverage"  # UNCERTAIN + INCOMPLETE_COVERAGE
    )
    # An offline state missing the proposed edge. Since missing graph evidence never
    # refutes, this is UNCERTAIN + INCOMPLETE_COVERAGE like the runtime-evolved
    # case; the two are kept apart because they say different things about how the
    # state entered the model, not because they are treated differently.
    OFFLINE_MISSING_EDGE = "offline_missing_edge"


class TopologyContinuationOutcome(BaseModel):
    """Pure classification of a topology-gap action proposal (never executes).

    ``refusal_code`` is the machine-readable form of ``details`` for every non-authorized
    outcome (``REJECTED`` and ``CONFIRMATION_REQUIRED``); it is ``None`` exactly when the
    outcome is ``AUTHORIZED_AUTO``.
    """

    authorization: TopologyAuthorization
    source_kind: TopologySourceKind | None = None
    reason: VerifyReason | None = None
    details: str = ""
    refusal_code: ContinuationRefusal | None = None
    current_state_id: str | None = None  # None for a novel/unlocalized source
    bound_action: dict[str, Any] | None = None


class TopologyContinuationClassifier:
    """Non-executing analog of :class:`SemanticGapContinuationClassifier` for topology gaps.

    It authorizes at most one unverified, evidence-gathering execution for a *topology*
    coverage gap — a novel screen or a runtime-evolved state missing an outgoing edge —
    but only when the proposed action is safe by the shared action-type policy
    (:func:`is_auto_eligible`). It never executes, mutates the EFSM, or calls an LLM; the
    outer policy (adapter) turns an ``AUTHORIZED_AUTO`` outcome into a receipt.
    """

    def __init__(
        self,
        fsm: AppFSM,
        engine: RuntimeEvolutionEngine,
    ) -> None:
        self._fsm = fsm
        self._engine = engine
        self._binder = ActionBinder()

    def classify(
        self,
        verification: VerificationOutput,
        screen: RawScreen,
        proposed_action: dict[str, Any] | ProposedAction,
    ) -> TopologyContinuationOutcome:
        """Return a topology authorization for a topology-shaped verifier verdict only."""
        verifier_kind = self._verifier_source_kind(verification)
        if verifier_kind is None:
            return self._rejected(
                verification,
                "verifier verdict is not a topology coverage gap",
                code=ContinuationRefusal.NOT_TOPOLOGY_GAP,
            )

        binding = self._binder.bind(proposed_action, screen)
        if binding.status is not BindingStatus.MATCH:
            return self._rejected(
                verification,
                f"action binding is {binding.status.value}, not match",
                code=ContinuationRefusal.ACTION_UNBOUND,
            )
        bound_action = binding.bound_action

        eligible, requires_confirmation, reason, source_code = classify_continuation_source(
            self._fsm, self._engine, screen, bound_action
        )
        if not eligible:
            return self._rejected(
                verification,
                reason,
                bound_action=bound_action,
                code=source_code or ContinuationRefusal.SOURCE_NOT_A_GAP,
            )

        # Cross-check: the live observation must still agree with the verifier's verdict.
        # A race (verifier said STATE_UNKNOWN, but the screen now localizes) is rejected.
        # This is load-bearing at execution time: the adapter's receipt revalidation
        # re-runs classify() with a receipt-derived verification and a *fresh* screen.
        observed_kind, current_state_id = self._observed_source_kind(screen)
        if observed_kind is not verifier_kind:
            return self._rejected(
                verification,
                "source classification no longer matches the verifier verdict",
                bound_action=bound_action,
                code=ContinuationRefusal.SOURCE_VERDICT_RACE,
            )

        # Automatic authorization is limited to the UNCERTAIN topology reasons, an
        # action that is safe by type alone, and a source the shared policy did not
        # flag as confirmation-required. A proven DENY and component actions stay
        # blocked. OFFLINE_MISSING_EDGE is on the list because missing evidence never
        # refutes, so a missing edge is always UNCERTAIN and there is nothing decisive
        # to self-authorize around. What still blocks a side effect is
        # is_auto_eligible, not the verdict.
        auto = (
            is_conservative_refusal(verification)
            and verifier_kind
            in (
                TopologySourceKind.NOVEL_UNLOCALIZED,
                TopologySourceKind.RUNTIME_EVOLVED_INCOMPLETE_COVERAGE,
                TopologySourceKind.OFFLINE_MISSING_EDGE,
            )
            and not requires_confirmation
            and is_auto_eligible(bound_action)
        )
        authorization = (
            TopologyAuthorization.AUTHORIZED_AUTO
            if auto
            else TopologyAuthorization.CONFIRMATION_REQUIRED
        )
        details = (
            "auto-eligible action on a topology coverage gap"
            if auto
            else "topology coverage gap requires confirmation and has no live surface"
        )
        # Which of the two confirmation conditions applies is the diagnostic difference
        # between "this action type is never auto-authorized" and "this source needs an
        # explicit confirmation channel"; both stay blocked.
        refusal_code = (
            None
            if auto
            else ContinuationRefusal.ACTION_NOT_AUTO_ELIGIBLE
            if not is_auto_eligible(bound_action)
            else ContinuationRefusal.CONFIRMATION_REQUIRED
        )
        return TopologyContinuationOutcome(
            authorization=authorization,
            source_kind=verifier_kind,
            reason=verification.reason,
            details=details,
            refusal_code=refusal_code,
            current_state_id=current_state_id,
            bound_action=bound_action,
        )

    def _verifier_source_kind(
        self,
        verification: VerificationOutput,
    ) -> TopologySourceKind | None:
        if not is_conservative_refusal(verification):
            return None
        if verification.reason is VerifyReason.STATE_UNKNOWN:
            return TopologySourceKind.NOVEL_UNLOCALIZED
        if verification.reason is not VerifyReason.INCOMPLETE_COVERAGE:
            return None
        # Both missing-edge sources carry this reason. Which one applies is a
        # property of the state, not of the verdict, so it is resolved from the
        # FSM here exactly as ``_observed_source_kind`` resolves it from the
        # observation -- otherwise the cross-check below would always race.
        state_id = verification.current_state_id
        state = self._fsm.states.get(state_id) if state_id is not None else None
        if state is not None and state.evidence.construction_source == RUNTIME_EVOLUTION_SOURCE:
            return TopologySourceKind.RUNTIME_EVOLVED_INCOMPLETE_COVERAGE
        return TopologySourceKind.OFFLINE_MISSING_EDGE

    def _observed_source_kind(
        self, screen: RawScreen
    ) -> tuple[TopologySourceKind | None, str | None]:
        classification = self._engine.classify_observation(screen)
        if classification.kind is ObservationClass.NOVEL:
            return TopologySourceKind.NOVEL_UNLOCALIZED, None
        if classification.kind is ObservationClass.EXISTING and classification.state_id:
            state = self._fsm.states[classification.state_id]
            if state.evidence.construction_source == RUNTIME_EVOLUTION_SOURCE:
                return (
                    TopologySourceKind.RUNTIME_EVOLVED_INCOMPLETE_COVERAGE,
                    classification.state_id,
                )
            return TopologySourceKind.OFFLINE_MISSING_EDGE, classification.state_id
        return None, None

    @staticmethod
    def _rejected(
        verification: VerificationOutput,
        details: str,
        *,
        bound_action: dict[str, Any] | None = None,
        code: ContinuationRefusal = ContinuationRefusal.NOT_TOPOLOGY_GAP,
    ) -> TopologyContinuationOutcome:
        return TopologyContinuationOutcome(
            authorization=TopologyAuthorization.REJECTED,
            source_kind=None,
            reason=verification.reason,
            details=details,
            refusal_code=code,
            current_state_id=verification.current_state_id,
            bound_action=bound_action,
        )


# --- post-action witness classification -----------------------------------


class PostActionWitnessShape(StrEnum):
    """Which post-action observation shape makes an executed step a witness.

    The names describe the *evidence*, not the operation the engine will pick: the
    engine still decides between ``abstraction_refinement``, ``coverage_expansion``,
    ``evidence_enrichment`` and a rejection from the observation itself.
    """

    DETERMINISM_CONFLICT_SPLIT = "determinism_conflict_split"
    NOVEL_SUCCESSOR_COVERAGE_EXPANSION = "novel_successor_coverage_expansion"
    INVARIANT_REFUTED_TARGET_SPLIT = "invariant_refuted_target_split"
    INVARIANT_REFUTED_SOURCE_SPLIT = "invariant_refuted_source_split"


class PostActionWitness(BaseModel):
    """Classification of one executed ``<pre, action, post>`` post-action observation.

    Exactly one of ``shape``/``refusal_code`` is set. ``split_subject_state_id`` names
    the state a refinement would partition, for the two invariant-refuted shapes and the
    determinism conflict; it is ``None`` for a novel successor.
    """

    shape: PostActionWitnessShape | None = None
    refusal_code: ContinuationRefusal | None = None
    details: str = ""
    source_state_id: str | None = None
    observed_state_id: str | None = None
    expected_target_state_id: str | None = None
    split_subject_state_id: str | None = None

    @property
    def is_witness(self) -> bool:
        """True when this observation may be submitted to the evolution engine."""
        return self.shape is not None


def refinement_identity_class(screen: RawScreen) -> str:
    """Refinement-identity class of one observation (the split discriminator)."""
    return refinement_identity_hash(raw_screen_to_quotient_input(screen))


def _attribution_refusal(
    fsm: AppFSM,
    raw_screens: dict[str, RawScreen],
    state_id: str,
    witness_screen: RawScreen,
) -> tuple[ContinuationRefusal | None, str]:
    """Refuse unless the witness screen carries a refinement class the state lacks.

    This is the same discriminator ``RuntimeEvolutionEngine._plan_source_split`` uses, so
    an observation that cannot possibly produce a split is refused here with a readable
    reason instead of deep inside the planner. Any unresolvable evidence observation is
    a refusal, never an assumption that the class is new.
    """
    state = fsm.states.get(state_id)
    if state is None:
        return (
            ContinuationRefusal.WITNESS_ATTRIBUTION_UNAVAILABLE,
            f"state {state_id} is not in the EFSM",
        )
    evidence_ids = list(state.evidence.raw_screen_ids)
    if not evidence_ids:
        return (
            ContinuationRefusal.WITNESS_ATTRIBUTION_UNAVAILABLE,
            f"state {state_id} has no resolvable evidence observation",
        )
    classes: set[str] = set()
    for screen_id in evidence_ids:
        screen = raw_screens.get(screen_id)
        if screen is None:
            return (
                ContinuationRefusal.WITNESS_ATTRIBUTION_UNAVAILABLE,
                f"evidence observation {screen_id} of {state_id} is unavailable",
            )
        classes.add(refinement_identity_class(screen))
    if refinement_identity_class(witness_screen) in classes:
        return (
            ContinuationRefusal.WITNESS_CLASS_ALREADY_COVERED,
            f"witness observation is in an existing refinement class of {state_id}",
        )
    return None, ""


def classify_post_action_witness(
    fsm: AppFSM,
    raw_screens: dict[str, RawScreen],
    *,
    pre_screen: RawScreen,
    pre_state_id: str,
    post_screen: RawScreen,
    post_output: VerificationOutput,
    observed_state_id: str | None,
    expected_target_state_id: str | None,
) -> PostActionWitness:
    """Classify one completed post-action check as a witness or a typed refusal.

    Runs *after* the evidence-only post-action verification, over the same observations,
    and never re-decides that verdict. It is pure: no EFSM mutation, no execution, no
    LLM. The caller (which owns policy and evidence persistence) submits an accepted
    witness to :func:`~vigil.runtime.observed_transition.apply_observed_transition`.

    Args:
        fsm: The app's EFSM as loaded for this check.
        raw_screens: Hydrated raw-screen evidence store (exploration trace + persisted
            runtime captures). Only read, for refinement-class attribution.
        pre_screen: The observation the executed action was verified against.
        pre_state_id: State ``pre_screen`` localized to at execution time.
        post_screen: The landed observation ``o_{t+1}``.
        post_output: Verdict returned by ``RuntimeMonitor.verify_post_action``.
        observed_state_id: State ``post_screen`` localized to (``None`` if unlocalized).
        expected_target_state_id: Successor the verified transition pointed at.

    Returns:
        A :class:`PostActionWitness` with either a ``shape`` or a ``refusal_code``.
    """
    base = {
        "source_state_id": pre_state_id,
        "observed_state_id": observed_state_id,
        "expected_target_state_id": expected_target_state_id,
    }

    # (a) The executed action had a modeled successor and landed in a different KNOWN
    #     state: the canonical action is nondeterministic at the source.
    if (
        post_output.proven_refutation
        and post_output.reason is VerifyReason.TRANSITION_INVALID
        and observed_state_id is not None
        and expected_target_state_id is not None
        and observed_state_id != expected_target_state_id
    ):
        return PostActionWitness(
            shape=PostActionWitnessShape.DETERMINISM_CONFLICT_SPLIT,
            details=(
                f"executed action reached {observed_state_id}, not the modeled successor "
                f"{expected_target_state_id}"
            ),
            split_subject_state_id=pre_state_id,
            **base,
        )

    # (c-target) Landed on the modeled successor, but one of its invariants is
    #     concretely false on the landed screen.
    if post_output.proven_refutation and post_output.reason is VerifyReason.INVARIANT_FAILED:
        subject = post_output.target_state_id or expected_target_state_id or observed_state_id
        if subject is None:
            return PostActionWitness(
                refusal_code=ContinuationRefusal.WITNESS_ATTRIBUTION_UNAVAILABLE,
                details="refuted invariant has no owning state",
                **base,
            )
        refusal, why = _attribution_refusal(fsm, raw_screens, subject, post_screen)
        if refusal is not None:
            return PostActionWitness(refusal_code=refusal, details=why, **base)
        return PostActionWitness(
            shape=PostActionWitnessShape.INVARIANT_REFUTED_TARGET_SPLIT,
            details=f"landed screen refutes an invariant of {subject}",
            split_subject_state_id=subject,
            **base,
        )

    if (
        is_conservative_refusal(post_output)
        and post_output.reason is VerifyReason.INVARIANT_INCONCLUSIVE
    ):
        return PostActionWitness(
            refusal_code=ContinuationRefusal.INVARIANT_INCONCLUSIVE,
            details="target invariant is inconclusive on the landed screen",
            **base,
        )

    # (b) The landing is outside the model. Only a NOVEL in-app observation is an
    #     admissible successor; an ambiguous identity, a foreign package, or a broken
    #     capture is a refusal — and, until now, an invisible one.
    if is_conservative_refusal(post_output) and post_output.reason in (
        VerifyReason.STATE_UNKNOWN,
        VerifyReason.STATE_SIMILAR,
    ):
        classification = RuntimeEvolutionEngine(fsm, {}).classify_observation(post_screen)
        if classification.kind is ObservationClass.NOVEL:
            return PostActionWitness(
                shape=PostActionWitnessShape.NOVEL_SUCCESSOR_COVERAGE_EXPANSION,
                details="executed action reached a novel in-app screen",
                **base,
            )
        if classification.kind in _OBSERVATION_REFUSALS:
            return PostActionWitness(
                refusal_code=_OBSERVATION_REFUSALS[classification.kind],
                details=(
                    f"landed observation is {classification.kind.value}, not an admissible "
                    "app state"
                ),
                **base,
            )
        return PostActionWitness(
            refusal_code=ContinuationRefusal.POST_LOCALIZATION_INCONCLUSIVE,
            details=(
                "post-action localization and live classification disagree; no stable witness"
            ),
            **base,
        )

    # (c-source) The successor was confirmed, but the *source* observation refutes an
    #     invariant of the state it localized to: the pre-screen belongs to a refinement
    #     class the source state does not cover. This is the shape that survives a
    #     conservative post-action check and still proves the abstraction is too coarse.
    source_invariants = DecisionEngine(fsm).check_state_invariants(pre_state_id, pre_screen)
    if (
        source_invariants.proven_refutation
        and source_invariants.reason is VerifyReason.INVARIANT_FAILED
    ):
        refusal, why = _attribution_refusal(fsm, raw_screens, pre_state_id, pre_screen)
        if refusal is not None:
            return PostActionWitness(refusal_code=refusal, details=why, **base)
        return PostActionWitness(
            shape=PostActionWitnessShape.INVARIANT_REFUTED_SOURCE_SPLIT,
            details=f"source observation refutes an invariant of {pre_state_id}",
            split_subject_state_id=pre_state_id,
            **base,
        )

    return PostActionWitness(
        refusal_code=ContinuationRefusal.NO_WITNESS,
        details="post-action check confirms the modeled successor; nothing to learn",
        **base,
    )

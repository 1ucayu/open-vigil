"""Conservative runtime EFSM evolution for unknown GUI states.

When runtime localization fails, Vigil does **not** guess topology from structural
similarity. Instead it treats an unknown observation as *candidate evidence*: the
continuation layer may execute a risk-bounded fallback and hand the resulting
``<pre, action, post>`` trace to this engine. The engine, at an action boundary,
performs *deterministic symbolic repair*:

- a novel screen may be admitted as a **PARTIAL** ``AbstractState`` (identity from the
  existing canonical/structural hashes), never from similarity;
- a real observed transition may be admitted as a new edge
  (origin provenance ``runtime_fallback``), never by copying a
  similar state's edges or guards.

For an already-known transition whose typed guard is missing, a separate targeted path
accepts ``<pre, admitted planner proposal/snapshot, bound action, optional post>``
semantic-gap evidence. It may revise only the admitted guard contract and the monotonic
EFSM revision; it never changes topology, action identity, edge trust,
transition provenance, observation counts, or target-state invariants.

Trust boundary (see project rules):

- Trust is binary and structural: membership in the transition graph is the whole of it, and
  runtime-fallback origin provenance is bookkeeping, never a gate. What keeps a
  runtime-created edge from ALLOWing on re-encounter is its admission state: the
  engine stamps every fresh runtime edge with a PENDING placeholder contract
  (:func:`vigil.models.guard.runtime_pending_placeholder`), and the origin-agnostic
  present-but-pending rule in ``DecisionEngine`` answers UNCERTAIN for it.
- An admitted runtime edge therefore stays UNCERTAIN until LLM guard/invariant
  candidates pass the existing deterministic offline admission gates (see
  :meth:`RuntimeEvolutionEngine.enrich_transition_contract`, or the between-round
  offline pipeline, either of which discharges the placeholder). The LLM only
  proposes typed contracts; it never decides topology, identity, or verdicts.
  Repeated fallback observations are corroborating trace evidence; they add
  provenance and observation counts but never discharge the placeholder.
- Similarity/templates may only help *retrieve* a candidate for diagnostics; they never
  justify adding a state, edge, or guard.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from vigil.core.ui_selectors import selector_has_stable_identity
from vigil.models.action import ActionType
from vigil.models.evolution import EvolutionOperation, EvolutionOutcome
from vigil.models.fsm import (
    FALLBACK_PROVENANCE_SOURCE,
    RUNTIME_EVOLUTION_SOURCE,
    AbstractState,
    AndroidStateContext,
    AppFSM,
    HierarchyLevel,
    ProvenanceEntry,
    StateEvidence,
    StateIdentity,
    StateInvariant,
    Transition,
    TransitionLookupStatus,
    canonical_action_key,
    record_efsm_revision,
    record_evolution_audit,
    transition_identity_key,
)
from vigil.models.guard import (
    GuardAdmissionStatus,
    GuardClause,
    GuardContract,
    is_runtime_pending_placeholder,
    runtime_pending_placeholder,
)
from vigil.models.intent_binding import AdmittedPlannerProposal
from vigil.models.state import RawScreen
from vigil.neuro.behavioral_signature import refinement_identity_hash
from vigil.neuro.guard_admission import admit_guard_contract, apply_guard_admission
from vigil.neuro.guard_contract_llm import generate_llm_guard_candidate
from vigil.neuro.guard_evidence import (
    GuardEvidence,
    ResolvedActionSemanticSurface,
    build_guard_evidence_for_transition,
    resolve_action_semantic_surface,
)
from vigil.neuro.guard_state_admission import check_slot_fingerprint_coherence
from vigil.neuro.prompt_redaction import build_prompt_redactor
from vigil.runtime.action_binder import ActionBinder, BindingStatus
from vigil.symbolic.dsl_evaluator import DSLEvaluator, GuardStatus
from vigil.symbolic.intent_patch import bound_value_error, intent_slot_fingerprint
from vigil.symbolic.invariant_checker import InvariantChecker
from vigil.symbolic.numeric_normalizer import (
    NumericNormalizationError,
    normalize_numeric_scalar,
)
from vigil.symbolic.screen_context import (
    runtime_elements_from_observation,
    screen_context_from_observation,
)
from vigil.symbolic.state_locator import (
    StateLocator,
    is_known_location,
    raw_screen_canonical_equivalence_hash,
    raw_screen_to_quotient_input,
)

# Global action types have GUI-control identity without a target selector.
_GLOBAL_ACTION_TYPES = frozenset({"navigate_back", "navigate_home"})


class ObservationClass(StrEnum):
    """Deterministic classification of a runtime observation."""

    EXISTING = "existing"  # canonically equivalent to a known state
    NOVEL = "novel"  # in-app, unclassifiable to any known state
    EXTERNAL = "external"  # different app package — not part of this EFSM
    AMBIGUOUS = "ambiguous"  # identity matches multiple known states
    UNCLASSIFIABLE = "unclassifiable"  # empty/broken capture


class ObservationClassification(BaseModel):
    """Result of :meth:`RuntimeEvolutionEngine.classify_observation`."""

    kind: ObservationClass
    state_id: str | None = None
    canonical_hash: str = ""


class ObservedFallbackStep(BaseModel):
    """One complete observed fallback step: the real ``<pre, action, post>`` triple.

    Holds concrete :class:`RawScreen` objects (not just ids) so guard/invariant
    evidence can be built for later semantic admission. In-memory only — never
    serialized into ``AppFSM``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pre_screen: RawScreen
    bound_action: dict[str, Any]
    post_screen: RawScreen


class AdmissionResult(BaseModel):
    """Outcome of a state/transition admission attempt."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    admitted: bool
    created: bool = False
    refined: bool = False
    state_id: str | None = None
    transition: Transition | None = None
    reason: str = ""


class PromotionResult(BaseModel):
    """Outcome of one LLM candidate → deterministic admission promotion pass."""

    model_config = ConfigDict(extra="forbid")

    guard_admitted: bool = False
    guard_admission_reason: str = ""
    guard_candidate_parsed_ok: bool = False
    guard_candidate_rejection_reason: str = ""
    guard_admission_status: str = "not_run"
    guard_admission_rejected_predicates: list[str] = Field(default_factory=list)
    guard_changed: bool = False
    guard_evidence_changed: bool = False
    guard_incomplete: bool = False


class SemanticGapEvidencePacket(BaseModel):
    """Planner/execution evidence for one existing-transition semantic gap.

    The admitted proposal freezes both the mobile-agent intent snapshot and the EFSM
    revision seen by that proposal. Vigil derives the resolved target and
    nearest stable ancestor semantic surface from ``pre_screen`` plus ``bound_action``;
    an optional post-screen is evidence only and never creates or corroborates an edge.
    """

    model_config = ConfigDict(extra="forbid")

    pre_screen: RawScreen
    proposal: AdmittedPlannerProposal
    bound_action: dict[str, Any]
    post_screen: RawScreen | None = None


class ContractEnrichmentResult(BaseModel):
    """Outcome of Contract enrichment for one existing transition."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    evidence_recorded: bool = False
    evolved: bool = False
    outcome: EvolutionOutcome = EvolutionOutcome.NOOP
    source_state_id: str | None = None
    target_state_id: str | None = None
    transition_index: int | None = None
    resolved_target: ResolvedActionSemanticSurface | None = None
    efsm_revision_before: int = 0
    efsm_revision_after: int = 0
    promotion: PromotionResult | None = None
    reason: str = ""


@dataclass
class _StateSplitPlan:
    """A validated, ready-to-commit Abstraction refinement (state split).

    Built entirely off to the side from a deep-copied view; only ``_commit_state_split``
    installs it into the live EFSM, so a rejected plan leaves the EFSM unchanged.
    """

    source_id: str
    states: dict[str, AbstractState]  # retained (keeps source_id) + refined siblings
    transitions: list[Transition]  # complete rebuilt labeled transition relation
    new_edge: Transition  # the newly observed conflicting edge, from its child


class RuntimeEvolutionEngine:
    """Conservative runtime EFSM evolution.

    Args:
        fsm: The app's FSM. Mutated in place only via ``AppFSM.add_state`` /
            ``add_transition`` (add a PARTIAL state / a runtime-origin edge).
        raw_screens: The existing raw-screen evidence store. Screens captured during
            fallback are registered here so guard/invariant evidence can be built.
    """

    def __init__(
        self,
        fsm: AppFSM,
        raw_screens: dict[str, RawScreen],
    ) -> None:
        self._fsm = fsm
        self._raw_screens = raw_screens
        # Concrete endpoint observations and executed fallback steps remain outside
        # AppFSM until deterministic admission.
        self._pending_screens: list[RawScreen] = []
        self._pending_steps: list[ObservedFallbackStep] = []
        self._capture_sequence = 0
        self._binder = ActionBinder()

    # -- classification --------------------------------------------------------

    def classify_observation(self, screen: RawScreen) -> ObservationClassification:
        """Deterministically classify a runtime observation.

        EXISTING when it canonically localizes to a known state; EXTERNAL when it
        belongs to a different app package; UNCLASSIFIABLE for an empty/broken
        capture; otherwise NOVEL. Similarity/template retrieval never changes the
        class.
        """
        if not screen.elements:
            return ObservationClassification(kind=ObservationClass.UNCLASSIFIABLE)

        # A per-app EFSM requires positive foreground-package evidence. Missing package
        # context is not proof that the screen belongs to the app.
        if not screen.package_name or not self._fsm.app_package:
            return ObservationClassification(kind=ObservationClass.UNCLASSIFIABLE)
        if screen.package_name != self._fsm.app_package:
            return ObservationClassification(kind=ObservationClass.EXTERNAL)

        canonical_hash = raw_screen_canonical_equivalence_hash(screen)
        location = StateLocator(self._fsm).locate(screen)
        if is_known_location(location.result) and location.state_id is not None:
            return ObservationClassification(
                kind=ObservationClass.EXISTING,
                state_id=location.state_id,
                canonical_hash=canonical_hash,
            )
        if self._has_ambiguous_identity(screen, canonical_hash):
            return ObservationClassification(
                kind=ObservationClass.AMBIGUOUS,
                canonical_hash=canonical_hash,
            )
        return ObservationClassification(kind=ObservationClass.NOVEL, canonical_hash=canonical_hash)

    # -- evidence recording (never mutates the transition graph) --------------

    def record_fallback_step(
        self,
        pre_screen: RawScreen,
        bound_action: dict[str, Any],
        post_screen: RawScreen,
    ) -> ObservedFallbackStep:
        """Record one executed fallback step as candidate evidence.

        Registers both screens in the raw-screen store, classifies both endpoints
        independently (a known source may transition to a novel target), indexes any
        novel endpoint as a candidate screen, and logs a ``FALLBACK_EXECUTION`` event.
        **Never** mutates ``states``/``transitions`` — admission is a separate step.
        """
        reserved_ids: set[str] = set()
        pre_snapshot = self._snapshot_screen(pre_screen, reserved_ids)
        post_snapshot = self._snapshot_screen(post_screen, reserved_ids)
        step = ObservedFallbackStep(
            pre_screen=pre_snapshot,
            bound_action=dict(bound_action),
            post_screen=post_snapshot,
        )
        endpoint_classes: list[ObservationClassification] = []
        for screen in (pre_snapshot, post_snapshot):
            classification = self.classify_observation(screen)
            endpoint_classes.append(classification)
            if classification.kind in {
                ObservationClass.EXISTING,
                ObservationClass.NOVEL,
            }:
                self._raw_screens[screen.screen_id] = screen
                self._pending_screens.append(screen)

        if all(
            classification.kind in {ObservationClass.EXISTING, ObservationClass.NOVEL}
            for classification in endpoint_classes
        ):
            self._pending_steps.append(step)

        self._fsm.evolution_log.append(
            {
                "event": "FALLBACK_EXECUTION",
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "pre_screen_id": pre_snapshot.screen_id,
                "post_screen_id": post_snapshot.screen_id,
                "action_type": bound_action.get("type"),
                "canonical_action_key": str(canonical_action_key(bound_action)),
            }
        )
        return step

    def pending_steps(self) -> list[ObservedFallbackStep]:
        """Executed in-app fallback steps awaiting transition admission."""
        return list(self._pending_steps)

    # -- existing-transition Contract enrichment ----------------------------

    def enrich_transition_contract(
        self,
        packet: SemanticGapEvidencePacket,
        llm: Any,
    ) -> ContractEnrichmentResult:
        """Enrich the contract of one deterministically known transition.

        This is an explicit evolution-stage operation, not part of execution-time
        verification.  It records live semantic evidence independently of transition
        provenance, invokes the existing typed generation/admission pipeline, and bumps
        ``AppFSM.efsm_revision`` only when an admitted guard actually changes.
        Topology, canonical action identity, and observation counts remain untouched.
        """
        revision_before = self._fsm.efsm_revision
        base_result = ContractEnrichmentResult(
            efsm_revision_before=revision_before,
            efsm_revision_after=revision_before,
        )
        if packet.proposal.efsm_revision != revision_before:
            return base_result.model_copy(
                update={
                    "reason": (
                        "proposal EFSM revision is stale: "
                        f"{packet.proposal.efsm_revision} != {revision_before}"
                    )
                }
            )

        location = StateLocator(self._fsm).locate(packet.pre_screen)
        if not is_known_location(location.result) or location.state_id is None:
            return base_result.model_copy(
                update={
                    "reason": (
                        "Contract enrichment requires deterministic source localization; "
                        f"got {location.result.value}"
                    )
                }
            )

        validated_action, binding_reason = self._validate_bound_action(
            packet.pre_screen,
            packet.bound_action,
        )
        if validated_action is None:
            return base_result.model_copy(
                update={"reason": f"bound action is not admissible: {binding_reason}"}
            )

        lookup = self._fsm.resolve_transition(location.state_id, validated_action)
        if lookup.status is not TransitionLookupStatus.MATCH or lookup.transition is None:
            return base_result.model_copy(
                update={
                    "source_state_id": location.state_id,
                    "reason": (
                        "Contract enrichment requires one existing resolved transition: "
                        f"{lookup.status.value} {lookup.details}"
                    ).strip(),
                }
            )
        transition = lookup.transition
        transition_index = self._fsm.transitions.index(transition)
        target_id = transition.target
        base_result = base_result.model_copy(
            update={
                "source_state_id": location.state_id,
                "target_state_id": target_id,
                "transition_index": transition_index,
            }
        )

        if packet.post_screen is not None:
            post_location = StateLocator(self._fsm).locate(packet.post_screen)
            if not is_known_location(post_location.result) or post_location.state_id != target_id:
                return base_result.model_copy(
                    update={
                        "reason": (
                            "post-screen does not deterministically localize to the "
                            "known transition "
                            f"target {target_id!r}"
                        )
                    }
                )

        resolved_target = resolve_action_semantic_surface(
            packet.pre_screen,
            validated_action,
        )
        action_type = str(validated_action.get("type") or "")
        if action_type in _GLOBAL_ACTION_TYPES:
            resolved_target = None
        elif resolved_target is None:
            return base_result.model_copy(
                update={"reason": "component target has no stable semantic surface"}
            )
        base_result = base_result.model_copy(update={"resolved_target": resolved_target})

        reserved_ids: set[str] = set()
        pre_snapshot = self._snapshot_screen(packet.pre_screen, reserved_ids)
        post_snapshot = (
            self._snapshot_screen(packet.post_screen, reserved_ids)
            if packet.post_screen is not None
            else None
        )
        self._raw_screens[pre_snapshot.screen_id] = pre_snapshot
        if post_snapshot is not None:
            self._raw_screens[post_snapshot.screen_id] = post_snapshot

        preferred_target_ids = [post_snapshot.screen_id] if post_snapshot is not None else []
        guard_evidence = build_guard_evidence_for_transition(
            self._fsm,
            transition,
            transition_index,
            self._raw_screens,
            preferred_source_screen_ids=[pre_snapshot.screen_id],
            preferred_target_screen_ids=preferred_target_ids,
            evidence_action_override=validated_action,
            runtime_intent_feedback={
                "snapshot_revision": packet.proposal.binding_snapshot.revision,
                "values": packet.proposal.binding_snapshot.values.model_dump(mode="json"),
            },
            frozen_slot_fingerprints=dict(packet.proposal.binding_snapshot.slot_fingerprints),
        )
        if resolved_target is not None:
            surface_error = self._semantic_surface_evidence_error(
                guard_evidence.source_observations,
                resolved_target.surface_resource_id,
            )
            if surface_error:
                self._raw_screens.pop(pre_snapshot.screen_id, None)
                if post_snapshot is not None:
                    self._raw_screens.pop(post_snapshot.screen_id, None)
                return base_result.model_copy(update={"reason": surface_error})

        record_evolution_audit(
            self._fsm,
            EvolutionOperation.EVIDENCE_ENRICHMENT,
            EvolutionOutcome.APPLIED,
            "CONTRACT_EVIDENCE_RECORDED",
            proposal_id=packet.proposal.proposal_id,
            source_state_id=location.state_id,
            target_state_id=target_id,
            transition_index=transition_index,
            pre_screen_id=pre_snapshot.screen_id,
            post_screen_id=post_snapshot.screen_id if post_snapshot is not None else None,
            bound_action=validated_action,
            resolved_target=(
                resolved_target.model_dump(mode="json") if resolved_target is not None else None
            ),
        )

        old_guard = transition.guard
        old_contract = (
            transition.guard_contract.model_copy(deep=True)
            if transition.guard_contract is not None
            else None
        )
        old_guard_view = _guard_semantic_view(old_guard, old_contract)

        # Revision-neutral: enrich_transition_contract commits its own single
        # Contract-enrichment revision below.
        promotion = _promote_semantics_uncommitted(
            self._fsm,
            transition,
            self._raw_screens,
            llm,
            guard_evidence_override=guard_evidence,
        )
        new_guard_view = _guard_semantic_view(
            transition.guard,
            transition.guard_contract,
        )
        guard_changed = promotion.guard_admitted and new_guard_view != old_guard_view

        if not guard_changed:
            transition.guard = old_guard
            transition.guard_contract = old_contract
            reason = promotion.guard_admission_reason or "admitted guard semantics are unchanged"
            outcome = (
                EvolutionOutcome.NOOP
                if promotion.guard_admitted
                else EvolutionOutcome.INCONCLUSIVE
                if promotion.guard_incomplete
                else EvolutionOutcome.REJECTED
            )
            record_evolution_audit(
                self._fsm,
                EvolutionOperation.CONTRACT_ENRICHMENT,
                outcome,
                "CONTRACT_ENRICHMENT_ATTEMPT",
                proposal_id=packet.proposal.proposal_id,
                transition_index=transition_index,
                reason=reason,
            )
            return base_result.model_copy(
                update={
                    "evidence_recorded": True,
                    "outcome": outcome,
                    "promotion": promotion,
                    "reason": reason,
                }
            )

        record_efsm_revision(
            self._fsm,
            EvolutionOperation.CONTRACT_ENRICHMENT,
            "CONTRACT_ENRICHMENT_GUARD",
            proposal_id=packet.proposal.proposal_id,
            transition_index=transition_index,
            guard=transition.guard,
        )
        return base_result.model_copy(
            update={
                "evidence_recorded": True,
                "evolved": True,
                "outcome": EvolutionOutcome.APPLIED,
                "efsm_revision_after": self._fsm.efsm_revision,
                "promotion": promotion,
                "reason": "admitted guard semantics for subsequent planner proposals",
            }
        )

    # -- symbolic admission ----------------------------------------------------

    def admit_state(self, candidate_screen: RawScreen) -> AdmissionResult:
        """Admit a specific candidate screen (Coverage expansion).

        Canonically-equivalent to an existing state → append the concrete screen to
        that state's evidence, except: a pending-step *source* observation is left for
        transition admission to attach to its resolved state (so an unresolved
        conflict/refinement never strands an observation on the parent), and an
        observation that breaks a trusted invariant is not attached but routed to
        Abstraction refinement. Otherwise add a PARTIAL state whose identity is
        derived from the existing hybrid/structural/canonical hashes.
        """
        pending_screen = self._find_pending_screen(candidate_screen)
        if pending_screen is None:
            return AdmissionResult(
                admitted=False,
                reason="screen is not recorded fallback evidence",
            )

        classification = self.classify_observation(pending_screen)
        if classification.kind in {
            ObservationClass.EXTERNAL,
            ObservationClass.AMBIGUOUS,
            ObservationClass.UNCLASSIFIABLE,
        }:
            return AdmissionResult(
                admitted=False, reason=f"not admissible: {classification.kind.value}"
            )

        if classification.kind is ObservationClass.EXISTING and classification.state_id:
            state = self._fsm.states[classification.state_id]
            if self._is_pending_step_source(pending_screen):
                self._remove_pending_screen(pending_screen)
                return AdmissionResult(
                    admitted=True,
                    created=False,
                    state_id=classification.state_id,
                    reason="existing source observation deferred to transition admission",
                )
            invariant_status = self._invariant_status(state, pending_screen)
            if invariant_status == "false":
                return AdmissionResult(
                    admitted=False,
                    state_id=classification.state_id,
                    reason=(
                        "observation violates a trusted invariant; abstraction refinement required"
                    ),
                )
            if invariant_status == "unknown":
                return AdmissionResult(
                    admitted=False,
                    state_id=classification.state_id,
                    reason="existing invariant is inconclusive on the observation; left pending",
                )
            self._attach_state_observation(state, pending_screen.screen_id)
            self._remove_pending_screen(pending_screen)
            return AdmissionResult(
                admitted=True,
                created=False,
                state_id=classification.state_id,
                reason="appended evidence to canonically-equivalent state",
            )

        new_state = self._build_partial_state(pending_screen, classification.canonical_hash)
        self._fsm.add_state(new_state)
        self._remove_pending_screen(pending_screen)
        self._record_revision(
            EvolutionOperation.COVERAGE_EXPANSION,
            "COVERAGE_EXPANSION_STATE",
            state_id=new_state.state_id,
        )
        logger.info(f"Runtime-evolution admitted PARTIAL state {new_state.state_id}")
        return AdmissionResult(
            admitted=True,
            created=True,
            state_id=new_state.state_id,
            reason="partial state admitted",
        )

    def admit_transition(self, step: ObservedFallbackStep) -> AdmissionResult:
        """Admit an observed edge from a real fallback step (Coverage expansion).

        Requires an executed real pre/post trace, a unique canonical action binding
        with stable target identity, and admitted source & target states. A
        ``(source, canonical_action_key)`` that already reaches a *different* target,
        or a source observation that breaks a trusted invariant, routes to Abstraction
        refinement (state split) rather than a flat reject. A same-target match
        corroborates (provenance only). Otherwise a fresh edge is admitted and the
        source observation attached, carrying runtime-fallback origin provenance.
        """
        if not step.bound_action:
            return AdmissionResult(admitted=False, reason="missing bound action")

        validated_action, binding_reason = self._validate_bound_action(
            step.pre_screen, step.bound_action
        )
        if validated_action is None:
            return AdmissionResult(
                admitted=False,
                reason=f"action is not a unique stable binding: {binding_reason}",
            )

        locator = StateLocator(self._fsm)
        source_loc = locator.locate(step.pre_screen)
        target_loc = locator.locate(step.post_screen)
        if not (is_known_location(source_loc.result) and source_loc.state_id):
            return AdmissionResult(admitted=False, reason="source state not admitted")
        if not (is_known_location(target_loc.result) and target_loc.state_id):
            return AdmissionResult(admitted=False, reason="target state not admitted")

        source_id = source_loc.state_id
        target_id = target_loc.state_id
        new_key = canonical_action_key(validated_action)
        source_state = self._fsm.states[source_id]
        target_state = self._fsm.states[target_id]

        # The post-screen must satisfy the target state's trusted invariants, else the edge
        # would lead into a state whose invariants are false on the observation. Checked
        # before any topology decision (refine included) so no path admits such an edge.
        target_status = self._invariant_status(target_state, step.post_screen)
        if target_status != "true":
            if target_id == source_id and target_status == "false":
                # Self-loop: route to the atomic split, which maps both endpoints and
                # revalidates invariants per child without weakening Γ/I or edge trust.
                return self._refine_source_on_conflict(
                    step,
                    validated_action,
                    source_id,
                    target_id,
                    base_reason="post-screen violates the self-loop state's invariant",
                )
            return AdmissionResult(
                admitted=False,
                state_id=source_id,
                reason=(
                    "post-screen violates the target state's invariants; abstraction "
                    "refinement required"
                    if target_status == "false"
                    else "target state invariants are inconclusive on the post-screen; left pending"
                ),
            )

        same_key = [
            existing
            for existing in self._fsm.transitions
            if existing.source == source_id and canonical_action_key(existing.action) == new_key
        ]
        conflict = [existing for existing in same_key if existing.target != target_id]
        corroborate = next(
            (existing for existing in same_key if existing.target == target_id), None
        )
        invariant_status = self._invariant_status(source_state, step.pre_screen)

        if conflict or invariant_status == "false":
            base_reason = (
                "determinism conflict" if conflict else "observation violates a trusted invariant"
            )
            return self._refine_source_on_conflict(
                step, validated_action, source_id, target_id, base_reason=base_reason
            )
        if invariant_status == "unknown":
            # An inconclusive existing invariant leaves the observation pending, no mutation.
            return AdmissionResult(
                admitted=False,
                state_id=source_id,
                reason="existing invariant is inconclusive on the observation; left pending",
            )

        if corroborate is not None:
            self._attach_state_observation(source_state, step.pre_screen.screen_id)
            corroborate.observed_count += 1
            corroborate.provenance.append(self._fallback_provenance(step, validated_action))
            self._remove_pending_step(step)
            record_evolution_audit(
                self._fsm,
                EvolutionOperation.EVIDENCE_ENRICHMENT,
                EvolutionOutcome.CORROBORATED,
                "TRANSITION_CORROBORATED",
                transition=str(transition_identity_key(corroborate)),
                observed_count=corroborate.observed_count,
            )
            return AdmissionResult(
                admitted=True,
                created=False,
                transition=corroborate,
                state_id=source_id,
                reason="corroborated existing edge",
            )

        new_transition = Transition(
            source=source_id,
            target=target_id,
            action=dict(validated_action),
            guard=None,
            # A runtime-witnessed edge enters the graph with a PENDING placeholder
            # contract, never the neutral guard: re-encounter routes through the
            # origin-agnostic present-but-pending rule (UNCERTAIN) until semantic
            # admission discharges it. Growth alone must not create trust.
            guard_contract=runtime_pending_placeholder(),
            observed_count=1,
            provenance=[self._fallback_provenance(step, validated_action)],
        )
        self._fsm.add_transition(new_transition)
        self._attach_state_observation(source_state, step.pre_screen.screen_id)
        self._remove_pending_step(step)
        self._record_revision(
            EvolutionOperation.COVERAGE_EXPANSION,
            "COVERAGE_EXPANSION_TRANSITION",
            transition=str(transition_identity_key(new_transition)),
        )
        logger.info(
            f"Runtime-evolution admitted edge {source_id} -> {target_id}"
            f" ({step.bound_action.get('type')})"
        )
        return AdmissionResult(
            admitted=True, created=True, transition=new_transition, state_id=source_id
        )

    # -- Abstraction refinement (state split + transition redistribution) -------

    def _refine_source_on_conflict(
        self,
        step: ObservedFallbackStep,
        validated_action: dict[str, Any],
        source_id: str,
        new_target_id: str,
        *,
        base_reason: str,
    ) -> AdmissionResult:
        """Split ``source_id`` so the conflicting canonical action is deterministic again.

        The complete state split and transition redistribution are planned and
        validated off to the side, then committed atomically; any failure leaves the
        live EFSM unchanged and rejects the conflicting transition. Distinction uses
        only the stable secondary features the runtime locator recomputes.
        """
        plan, reason = self._plan_source_split(step, validated_action, source_id, new_target_id)
        if plan is None:
            return AdmissionResult(
                admitted=False,
                state_id=source_id,
                reason=f"{base_reason}; abstraction refinement not possible: {reason}",
            )
        self._commit_state_split(plan)
        self._remove_pending_step(step)
        self._record_revision(
            EvolutionOperation.ABSTRACTION_REFINEMENT,
            "ABSTRACTION_REFINEMENT",
            source_state=source_id,
            new_states=sorted(sid for sid in plan.states if sid != source_id),
            canonical_action_key=str(canonical_action_key(validated_action)),
        )
        logger.info(
            f"Runtime-evolution split state {source_id} into {sorted(plan.states)} "
            "on a canonical-action conflict"
        )
        return AdmissionResult(
            admitted=True,
            created=True,
            refined=True,
            transition=plan.new_edge,
            state_id=plan.new_edge.source,
            reason="abstraction refinement: state split and transition redistribution committed",
        )

    def _plan_source_split(
        self,
        step: ObservedFallbackStep,
        validated_action: dict[str, Any],
        source_id: str,
        new_target_id: str,
    ) -> tuple[_StateSplitPlan | None, str]:
        source_state = self._fsm.states[source_id]

        # Repeated refinement would lose a prior partition identity; reject conservatively.
        if source_state.identity.refinement_signature is not None or (
            source_state.identity.structural_hash
            and "::secondary:" in source_state.identity.structural_hash
        ):
            return None, "source state is already a refined state"

        template_reason = self._template_reference_reason(source_id)
        if template_reason:
            return None, template_reason

        new_source_screen = step.pre_screen.screen_id
        new_post_screen = step.post_screen.screen_id if new_target_id == source_id else None

        # Pre-existing verified evidence = state membership ∪ every old transition-endpoint
        # provenance observation of the source. Partition ALL source observations by the
        # deterministic, XML-derived refinement identity (Γ/I never defines the partition).
        affected = [
            t for t in self._fsm.transitions if t.source == source_id or t.target == source_id
        ]
        verified_evidence: set[str] = set(source_state.evidence.raw_screen_ids)
        for t in affected:
            for e in t.provenance:
                if t.source == source_id and e.source_screen_id:
                    verified_evidence.add(e.source_screen_id)
                if t.target == source_id and e.target_screen_id:
                    verified_evidence.add(e.target_screen_id)
        all_obs = set(verified_evidence) | {new_source_screen}
        if new_post_screen:
            all_obs.add(new_post_screen)

        identity_of: dict[str, str] = {}
        for sid in all_obs:
            rid = self._refinement_hash(sid)
            if rid is None:
                return None, f"source observation {sid} is unavailable for distinction"
            identity_of[sid] = rid

        children_screens: dict[str, set[str]] = defaultdict(set)
        for sid, rid in identity_of.items():
            children_screens[rid].add(sid)
        if len(children_screens) < 2:
            return None, "no schema/action-surface discriminator distinguishes the observations"

        # The pre-existing verified evidence must map to exactly one refinement identity;
        # that child keeps source_id, every other identity becomes a fresh sibling.
        verified_ids = {identity_of[sid] for sid in verified_evidence}
        if len(verified_ids) != 1:
            return None, "pre-existing verified evidence spans multiple refinement identities"
        retained_rid = next(iter(verified_ids))

        family_id = f"rf::{source_id}"
        app_package = self._fsm.app_package or ""
        child_of_rid: dict[str, str] = {}
        sibling_index = 0
        for rid in sorted(children_screens):
            if rid == retained_rid:
                child_of_rid[rid] = source_id
            else:
                sibling_index += 1
                child_of_rid[rid] = self._allocate_split_child_id(source_id, sibling_index)
        child_of_screen = {sid: child_of_rid[identity_of[sid]] for sid in all_obs}
        child_screens: dict[str, set[str]] = defaultdict(set)
        for sid, child in child_of_screen.items():
            child_screens[child].add(sid)

        # Build child states with evidence-derived identities (live-consistent, never the
        # offline hash) and a family-scoped refinement signature.
        new_states: dict[str, AbstractState] = {}
        for rid, child_id in child_of_rid.items():
            screens = sorted(child_screens[child_id])
            representative = self._raw_screens[screens[0]]
            if child_id == source_id:
                child_state = source_state.model_copy(deep=True)
            else:
                child_state = source_state.model_copy(
                    deep=True,
                    update={
                        "state_id": child_id,
                        "name": f"{source_state.name}__refined_{child_id.rsplit('_', 1)[-1]}",
                    },
                )
                child_state.evidence.construction_source = RUNTIME_EVOLUTION_SOURCE
            child_state.evidence.raw_screen_ids = screens
            child_state.identity.functional_hash = representative.get_hybrid_state_id(app_package)
            child_state.identity.structural_hash = representative.get_structural_fingerprint()
            child_state.identity.canonical_equivalence_hash = raw_screen_canonical_equivalence_hash(
                representative
            )
            child_state.identity.refinement_signature = rid
            child_state.identity.refinement_family = family_id
            # Per-child invariant revalidation: never weaken a verified invariant.
            revalidated, reason = self._revalidate_child_invariants(
                source_state.invariant_specs,
                child_screens[child_id],
                contains_verified=bool(child_screens[child_id] & verified_evidence),
            )
            if revalidated is None:
                return None, reason
            child_state.invariant_specs = revalidated
            new_states[child_id] = child_state

        # Redistribute affected transitions. With homogeneous verified evidence every
        # affected edge's endpoint observations stay in the retained child, so each is
        # preserved unchanged (guard, trust, observed_count intact); a would-be
        # relocation of a guarded/verified edge rejects the split.
        rebuilt: list[Transition] = []
        for t in self._fsm.transitions:
            if t.source != source_id and t.target != source_id:
                rebuilt.append(t)
                continue
            if not t.provenance:
                return None, (
                    f"affected edge {t.source}->{t.target} lacks provenance to redistribute"
                )
            groups: dict[tuple[str, str], list[ProvenanceEntry]] = defaultdict(list)
            for e in t.provenance:
                if t.source == source_id:
                    if not e.source_screen_id or e.source_screen_id not in child_of_screen:
                        return None, f"edge {t.source}->{t.target} has an unassignable source obs"
                    s_child = child_of_screen[e.source_screen_id]
                else:
                    s_child = t.source
                if t.target == source_id:
                    if not e.target_screen_id or e.target_screen_id not in child_of_screen:
                        return None, f"edge {t.source}->{t.target} has an unassignable target obs"
                    tgt_child = child_of_screen[e.target_screen_id]
                else:
                    tgt_child = t.target
                groups[(s_child, tgt_child)].append(e)
            partitioned = len(groups) > 1
            for (s_child, tgt_child), entries in groups.items():
                identity_changed = s_child != t.source or tgt_child != t.target
                preserve = not partitioned and not identity_changed
                # A pending runtime placeholder is not a verified guard: it carries
                # no admitted semantics to lose, so the edge relocates and gets a
                # fresh placeholder (still UNCERTAIN until admission). Only a real
                # analyzed contract blocks relocation.
                placeholder = is_runtime_pending_placeholder(t.guard_contract)
                if t.guard_contract is not None and not preserve and not placeholder:
                    return None, (
                        f"split would relocate guarded edge {t.source}->{t.target}; cannot "
                        "preserve its verified guard"
                    )
                rebuilt.append(
                    Transition(
                        source=s_child,
                        target=tgt_child,
                        action=dict(t.action),
                        guard=t.guard if preserve else None,
                        guard_contract=(
                            t.guard_contract.model_copy(deep=True)
                            if preserve and t.guard_contract is not None
                            else runtime_pending_placeholder()
                            if placeholder
                            else None
                        ),
                        # A changed or partitioned identity is not the edge that was
                        # observed, so this evidence cannot attribute it to exactly one
                        # sibling; it enters the graph marked unattributable for refinement.
                        ambiguous_evidence=(t.ambiguous_evidence if preserve else True),
                        observed_count=t.observed_count if preserve else len(entries),
                        provenance=[e.model_copy(deep=True) for e in entries],
                    )
                )

        # The newly observed conflicting/violating step becomes an explicit new edge;
        # a self-loop maps its post-screen to the post-screen's child.
        new_edge_target = (
            child_of_screen[new_post_screen] if new_post_screen is not None else new_target_id
        )
        new_edge = Transition(
            source=child_of_screen[new_source_screen],
            target=new_edge_target,
            action=dict(validated_action),
            guard=None,
            # Same rule as admit_transition's fresh edge: runtime-witnessed, so it
            # enters the graph with a PENDING placeholder and stays UNCERTAIN on
            # re-encounter until semantic admission discharges it.
            guard_contract=runtime_pending_placeholder(),
            observed_count=1,
            provenance=[self._fallback_provenance(step, validated_action)],
        )
        rebuilt.append(new_edge)

        # Initial state: raw_screen_ids are membership, not a launch witness, so the
        # initial state can never be split.
        if source_id == self._fsm.initial_state:
            return None, "cannot split the initial state without a unique launch witness"

        plan = _StateSplitPlan(
            source_id=source_id,
            states=new_states,
            transitions=rebuilt,
            new_edge=new_edge,
        )
        error = self._validate_split_plan(plan)
        if error:
            return None, error
        return plan, ""

    def _revalidate_child_invariants(
        self,
        inherited: list[StateInvariant],
        child_screen_ids: set[str],
        *,
        contains_verified: bool,
    ) -> tuple[list[StateInvariant] | None, str]:
        """Revalidate inherited invariants against one child's observations.

        A child holding pre-existing verified evidence must keep every invariant
        (``FALSE``/``UNKNOWN``/missing evidence → reject the split); a new-evidence-only
        child keeps ``TRUE`` invariants, omits only proven-``FALSE`` ones, and rejects on
        ``UNKNOWN``. Retained invariants get an idempotent child-scoped ``evidence_count``.
        """
        kept: list[StateInvariant] = []
        support = len(child_screen_ids)
        for spec in inherited:
            status = self._evaluate_invariant_over_screens(spec.expr, child_screen_ids)
            if status == "true":
                kept.append(spec.model_copy(update={"evidence_count": support}))
            elif contains_verified:
                return None, (f"child holding verified evidence would lose invariant {spec.expr!r}")
            elif status == "false":
                continue
            else:  # UNKNOWN / missing evidence on a new-evidence-only child
                return None, f"invariant {spec.expr!r} is inconclusive on a new refined child"
        return kept, ""

    def _validate_split_plan(self, plan: _StateSplitPlan) -> str:
        """Validate a split plan before any live mutation (Abstraction refinement)."""
        final_states = dict(self._fsm.states)
        final_states.update(plan.states)

        for t in plan.transitions:
            if t.source not in final_states or t.target not in final_states:
                return f"redistributed edge {t.source}->{t.target} has an invalid endpoint"

        # Determinism across all active transitions, however their evidence is marked.
        seen_target: dict[tuple[Any, ...], str] = {}
        for t in plan.transitions:
            action_key = canonical_action_key(t.action)
            existing_target = seen_target.get((t.source, action_key))
            if existing_target is not None and existing_target != t.target:
                return "state split leaves a nondeterministic canonical action"
            seen_target[(t.source, action_key)] = t.target

        # Provenance preservation: every original affected entry appears exactly once,
        # plus the single new-edge entry.
        child_ids = set(plan.states)
        original = sum(
            len(t.provenance)
            for t in self._fsm.transitions
            if t.source == plan.source_id or t.target == plan.source_id
        )
        redistributed = sum(
            len(t.provenance)
            for t in plan.transitions
            if t is not plan.new_edge and (t.source in child_ids or t.target in child_ids)
        )
        if redistributed != original:
            return "state split did not preserve every provenance entry exactly once"
        return ""

    def _commit_state_split(self, plan: _StateSplitPlan) -> None:
        """Install a validated split into the live EFSM in one atomic swap."""
        self._fsm.states.update(plan.states)
        self._fsm.transitions = plan.transitions
        self._fsm.rebuild_graph()

    # -- helpers ---------------------------------------------------------------

    def _is_pending_step_source(self, screen: RawScreen) -> bool:
        """True when ``screen`` is the source (pre-screen) of a still-pending step."""
        return any(step.pre_screen is screen for step in self._pending_steps)

    def _invariant_status(self, state: AbstractState, screen: RawScreen) -> str:
        """Three-valued status of a state's invariants against one observation.

        Returns ``"false"`` (a proven violation → route to Abstraction refinement),
        ``"unknown"`` (inconclusive → leave the observation pending, no mutation), or
        ``"true"`` (all invariants hold, or the state has none → proceed).
        """
        if not state.invariant_specs:
            return "true"
        screen_ctx = screen_context_from_observation(screen)
        result = InvariantChecker(self._fsm).check_state(state.state_id, screen_ctx)
        if result.failed_invariants:
            return "false"
        if result.unknown_invariants:
            return "unknown"
        return "true"

    def _evaluate_invariant_over_screens(self, expr: str, screen_ids: set[str]) -> str:
        """Three-valued evaluation of one invariant over a set of observations.

        ``"false"`` if it is proven false on any observation; ``"unknown"`` if any
        observation is missing or inconclusive (and none false); ``"true"`` only when it
        holds across every resolved observation.
        """
        evaluator = DSLEvaluator()
        saw_unknown = False
        for screen_id in screen_ids:
            screen = self._raw_screens.get(screen_id)
            if screen is None:
                saw_unknown = True
                continue
            status = evaluator.evaluate(
                expr, screen_ctx=screen_context_from_observation(screen)
            ).status
            if status is GuardStatus.FALSE:
                return "false"
            if status is GuardStatus.UNKNOWN:
                saw_unknown = True
        return "unknown" if saw_unknown else "true"

    @staticmethod
    def _attach_state_observation(state: AbstractState, screen_id: str) -> None:
        """Append a raw-screen observation to a state's evidence, deduplicated."""
        if screen_id not in state.evidence.raw_screen_ids:
            state.evidence.raw_screen_ids.append(screen_id)

    def _record_revision(
        self,
        operation: EvolutionOperation,
        event: str,
        **fields: Any,
    ) -> None:
        """Increment ``efsm_revision`` once via the single shared commit path."""
        record_efsm_revision(self._fsm, operation, event, **fields)

    def _allocate_state_id(self, canonical_hash: str) -> str:
        """Deterministically allocate a runtime state id, resolving collisions."""
        base = f"s_rt_{canonical_hash[:12]}"
        if base not in self._fsm.states:
            return base
        suffix = 1
        while f"{base}_{suffix}" in self._fsm.states:
            suffix += 1
        return f"{base}_{suffix}"

    def _allocate_split_child_id(self, source_id: str, index: int) -> str:
        """Deterministic refined-sibling id, resolving collisions."""
        base = f"{source_id}__refined_{index}"
        candidate = base
        collision = 0
        while candidate in self._fsm.states:
            collision += 1
            candidate = f"{base}_{collision}"
        return candidate

    def _template_reference_reason(self, source_id: str) -> str | None:
        """Reject splitting a state referenced by a sub-FSM template (first implementation)."""
        for template_id, template in self._fsm.sub_fsm_templates.items():
            if template.source_state_id == source_id or source_id in template.states:
                return f"state participates in sub-FSM template {template_id}"
            for edge in template.transitions:
                if edge.source == source_id or edge.target == source_id:
                    return f"state is referenced by sub-FSM template {template_id}"
        return None

    def _refinement_hash(self, screen_id: str) -> str | None:
        """Refinement identity of a stored runtime observation, or None if unavailable."""
        screen = self._raw_screens.get(screen_id)
        if screen is None:
            return None
        return refinement_identity_hash(raw_screen_to_quotient_input(screen))

    def _build_partial_state(self, screen: RawScreen, canonical_hash: str) -> AbstractState:
        app_package = self._fsm.app_package or ""
        structural_fp = screen.get_structural_fingerprint()
        functional_hash = screen.get_hybrid_state_id(app_package)
        state_id = self._allocate_state_id(canonical_hash)
        return AbstractState(
            state_id=state_id,
            name=f"runtime_partial_{canonical_hash[:8]}",
            hierarchy_level=HierarchyLevel.ACTIVITY,
            identity=StateIdentity(
                functional_hash=functional_hash,
                structural_hash=structural_fp,
                canonical_equivalence_hash=canonical_hash,
            ),
            android_context=AndroidStateContext(
                activity_name=screen.activity_name,
                package_name=screen.package_name,
            ),
            evidence=StateEvidence(
                raw_screen_ids=[screen.screen_id],
                construction_source=RUNTIME_EVOLUTION_SOURCE,
            ),
        )

    @staticmethod
    def _fallback_provenance(
        step: ObservedFallbackStep, bound_action: dict[str, Any]
    ) -> ProvenanceEntry:
        return ProvenanceEntry(
            trace_step_index=-1,
            source_screen_id=step.pre_screen.screen_id,
            target_screen_id=step.post_screen.screen_id,
            raw_action=dict(bound_action),
            evidence_source=FALLBACK_PROVENANCE_SOURCE,
        )

    def _validate_bound_action(
        self,
        screen: RawScreen,
        bound_action: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str]:
        """Re-prove a unique persistent action binding on one concrete screen."""
        raw_type = bound_action.get("type") or bound_action.get("action_type")
        if not isinstance(raw_type, str):
            return None, f"unsupported action type {raw_type!r}"
        try:
            action_type = ActionType(raw_type)
        except ValueError:
            return None, f"unsupported action type {raw_type!r}"

        if action_type.value in _GLOBAL_ACTION_TYPES:
            canonical_global = {"type": action_type.value}
            if canonical_action_key(bound_action) != canonical_action_key(canonical_global):
                return None, "global action carries component identity"
            return canonical_global, ""

        binding = self._binder.bind(bound_action, screen)
        if binding.status is not BindingStatus.MATCH:
            return None, f"fresh binding is {binding.status.value}: {binding.reason}"

        # Remove the capture-local handle and prove that the persistent selector alone
        # still resolves uniquely on the evidence screen.
        persistent_action = dict(binding.bound_action)
        persistent_action.pop("target", None)
        persistent_binding = self._binder.bind(persistent_action, screen)
        if persistent_binding.status is not BindingStatus.MATCH:
            return None, (
                "persistent selector binding is "
                f"{persistent_binding.status.value}: {persistent_binding.reason}"
            )
        selector = persistent_binding.bound_action.get("target_selector")
        if not isinstance(selector, dict) or not selector_has_stable_identity(selector):
            return None, "component action lacks a strong stable selector"
        if canonical_action_key(bound_action) != canonical_action_key(
            persistent_binding.bound_action
        ):
            return None, "recorded action was not fully bound to its evidence screen"
        return persistent_binding.bound_action, ""

    @staticmethod
    def _semantic_surface_evidence_error(
        observations: list[dict[str, Any]],
        surface_resource_id: str,
    ) -> str:
        """Require one readable containment surface in every source observation."""
        if not surface_resource_id:
            return "resolved semantic surface lacks runtime-resolvable resource identity"
        for observation in observations:
            screen_id = str(observation.get("screen_id") or "<unknown>")
            matches = [
                element
                for element in runtime_elements_from_observation(observation)
                if str(element.get("resource_id") or "") == surface_resource_id
            ]
            if len(matches) != 1:
                return (
                    f"semantic surface {surface_resource_id!r} resolves {len(matches)} times "
                    f"in source observation {screen_id}"
                )
            properties = screen_context_from_observation(observation).elements.get(
                surface_resource_id,
                {},
            )
            readable_values = [
                str(properties.get("text") or "").strip(),
                str(properties.get("content_description") or "").strip(),
                *[
                    str(child.get("text") or "").strip()
                    for child in properties.get("children", [])
                    if isinstance(child, dict)
                ],
            ]
            if not any(readable_values):
                return (
                    f"semantic surface {surface_resource_id!r} has no readable value "
                    f"in source observation {screen_id}"
                )
        return ""

    def _has_ambiguous_identity(self, screen: RawScreen, canonical_hash: str) -> bool:
        """True when deterministic identity maps the observation to multiple states."""
        candidates = {
            screen.get_structural_fingerprint(),
            screen.get_hybrid_state_id(self._fsm.app_package),
            screen.get_state_id(self._fsm.app_package),
        }
        for fingerprint in candidates:
            matches = {
                state.state_id
                for state in self._fsm.states.values()
                if fingerprint in {state.identity.functional_hash, state.identity.structural_hash}
            }
            if len(matches) > 1:
                return True
        canonical_matches = {
            state.state_id
            for state in self._fsm.states.values()
            if state.identity.canonical_equivalence_hash == canonical_hash
        }
        return len(canonical_matches) > 1

    def _snapshot_screen(self, screen: RawScreen, reserved_ids: set[str]) -> RawScreen:
        """Copy one capture under a unique evidence id without mutating the caller object."""
        base = screen.screen_id.strip() or "runtime_capture"
        screen_id = base
        while screen_id in self._raw_screens or screen_id in reserved_ids:
            self._capture_sequence += 1
            screen_id = f"{base}__rt_{self._capture_sequence:06d}"
        reserved_ids.add(screen_id)
        return screen.model_copy(update={"screen_id": screen_id}, deep=True)

    def _find_pending_screen(self, screen: RawScreen) -> RawScreen | None:
        for candidate in self._pending_screens:
            if candidate is screen:
                return candidate
        return None

    def _remove_pending_screen(self, screen: RawScreen) -> None:
        self._pending_screens = [
            candidate for candidate in self._pending_screens if candidate is not screen
        ]

    def _remove_pending_step(self, step: ObservedFallbackStep) -> None:
        self._pending_steps = [
            candidate for candidate in self._pending_steps if candidate is not step
        ]


# --- Contract enrichment (additive transition guards) ---------------------


@dataclass
class GuardEnrichmentResult:
    """Outcome of additive guard Contract enrichment on one transition."""

    changed: bool = False
    evidence_changed: bool = False
    admitted: bool = False
    reason: str = ""
    admission_status: str = "not_run"
    rejected_predicates: list[str] = field(default_factory=list)


def _guard_semantic_view(guard: str | None, contract: GuardContract | None) -> tuple[Any, Any]:
    """Normalized comparison view of a guard/contract for change detection."""
    if contract is None:
        return guard, None
    slots = sorted(
        (slot.name, slot.slot_type.value, slot.requiredness.value, tuple(slot.value_domain))
        for slot in contract.slots
    )
    clauses = sorted(
        (
            (clause.slot, clause.compiled_expr)
            for clause in contract.clauses
            if clause.compiled_expr
        ),
        key=lambda item: (item[0] or "", item[1]),
    )
    return guard, (
        contract.admission_status.value,
        contract.kind.value,
        tuple(slots),
        tuple(clauses),
    )


def _guard_predicate_identity(clause: GuardClause) -> tuple[Any, ...]:
    """Canonical typed predicate identity for pre-admission clause deduplication.

    Uses the typed predicate + slot, never ``compiled_expr`` (empty before admission).
    """
    predicate = clause.predicate
    expected = predicate.expected
    return (
        clause.slot,
        predicate.predicate_type,
        predicate.element,
        predicate.property,
        predicate.operator,
        None if expected is None else expected.kind,
        None if expected is None else expected.value,
        None if expected is None else expected.slot,
    )


def _merge_guard_contract(
    existing: GuardContract, candidate: GuardContract
) -> tuple[GuardContract | None, str]:
    """Additively merge candidate guard metadata into an existing admitted contract.

    Preserves every existing clause and the existing intent-slot schema; appends only
    clauses whose canonical typed predicate identity is new; unions slots, rejecting a
    conflicting redefinition of a slot name; **preserves the existing GuardKind** (a
    duplicate/other clause never relabels e.g. NAVIGATION as SAFETY_CHECK merely because
    the LLM proposed a different kind); and deduplicates semantic evidence. Returns
    ``(merged, "")`` or ``(None, reason)`` on an intent-slot conflict. The whole effective
    contract is re-admitted by the caller, so clause admission metadata is reset to pending.
    """
    merged = existing.model_copy(deep=True)

    slots_by_name = {slot.name: slot for slot in merged.slots}
    for slot in candidate.slots:
        prior = slots_by_name.get(slot.name)
        if prior is None:
            merged.slots.append(slot.model_copy(deep=True))
            slots_by_name[slot.name] = slot
            continue
        if prior.slot_type != slot.slot_type or sorted(prior.value_domain) != sorted(
            slot.value_domain
        ):
            return None, f"conflicting definition for intent slot {slot.name!r}"

    seen_predicates = {_guard_predicate_identity(clause) for clause in merged.clauses}
    for clause in candidate.clauses:
        identity = _guard_predicate_identity(clause)
        if identity in seen_predicates:
            continue
        merged.clauses.append(clause.model_copy(deep=True))
        seen_predicates.add(identity)

    seen_refs = {(ref.source, ref.identifier) for ref in merged.semantic_evidence}
    for ref in candidate.semantic_evidence:
        if (ref.source, ref.identifier) not in seen_refs:
            merged.semantic_evidence.append(ref.model_copy(deep=True))
            seen_refs.add((ref.source, ref.identifier))

    merged.kind = existing.kind
    merged.admission_status = GuardAdmissionStatus.PENDING
    merged.admission_reason = ""
    for clause in merged.clauses:
        clause.admission_status = GuardAdmissionStatus.PENDING
        clause.admission_reason = ""
        clause.compiled_expr = ""
    return merged, ""


def _validate_new_intent_clauses(
    old_contract: GuardContract | None,
    effective: GuardContract,
    guard_evidence: GuardEvidence,
) -> str | None:
    """All-or-nothing frozen-witness check for newly introduced intent-dependent clauses.

    A *new* intent clause (present in the effective contract but not the old one, detected
    via ``predicate.expected.kind == "intent"`` + ``expected.slot`` because a candidate
    clause's ``slot`` is unset before admission) must have a matching frozen witness: the
    slot name present in the snapshot, ``intent_slot_fingerprint`` equal to the frozen
    value, a ``bound`` status, and a value matching the slot type/enum. With no frozen
    witness at all, any new intent clause is rejected. Returns a reason on the first
    failure, else ``None``.
    """
    old_predicates = (
        {_guard_predicate_identity(clause) for clause in old_contract.clauses}
        if old_contract is not None
        else set()
    )
    slots_by_name = {slot.name: slot for slot in effective.slots}
    values = guard_evidence.runtime_intent_feedback.get("values", {})
    fingerprints = guard_evidence.frozen_slot_fingerprints
    for clause in effective.clauses:
        expected = clause.predicate.expected
        if expected is None or expected.kind != "intent" or not expected.slot:
            continue
        if _guard_predicate_identity(clause) in old_predicates:
            continue  # already-admitted clause, not newly introduced
        slot_name = expected.slot
        if not fingerprints:
            return f"new intent clause for slot {slot_name!r} has no frozen witness"
        slot = slots_by_name.get(slot_name)
        if slot is None:
            return f"new intent clause references undeclared slot {slot_name!r}"
        frozen = fingerprints.get(slot_name)
        if frozen is None or intent_slot_fingerprint(slot) != frozen:
            return f"new intent clause slot {slot_name!r} does not match the frozen fingerprint"
        if not isinstance(values, dict) or slot_name not in values:
            return f"new intent clause slot {slot_name!r} is not bound in the frozen snapshot"
        if bound_value_error(slot, values[slot_name]):
            return f"new intent clause slot {slot_name!r} has a mismatched bound value"
    return None


def _bound_runtime_values(guard_evidence: GuardEvidence) -> list[Any]:
    """Every value bound in the frozen intent snapshot carried by the live evidence.

    These are task-specific values the mobile agent resolved at runtime. They may back an
    ``$intent.*`` clause (validated by :func:`_validate_new_intent_clauses`) but must never
    be frozen into a literal predicate, which would silently pin one execution's value into
    the shared EFSM.
    """
    values = guard_evidence.runtime_intent_feedback.get("values", {})
    if not isinstance(values, dict):
        return []
    return list(values.values())


def _literal_copies_bound_value(literal_value: Any, operator: str | None, bound_value: Any) -> bool:
    """True when a literal predicate operand reproduces a runtime-bound value.

    Deterministic, using the same normalization as bound-value admission: numeric operands
    match on ``Decimal`` equality; string operands match on a casefolded/stripped equality,
    and for ``contains`` also when the bound string appears as a substring
    of the literal; other scalars match on plain equality.
    """
    if literal_value is None or bound_value is None:
        return False
    try:
        if normalize_numeric_scalar(literal_value, source_value=False) == normalize_numeric_scalar(
            bound_value, source_value=False
        ):
            return True
    except NumericNormalizationError:
        pass
    if isinstance(literal_value, str) and isinstance(bound_value, str):
        lit = literal_value.strip().casefold()
        bound = bound_value.strip().casefold()
        if not bound:
            return False
        if operator == "contains":
            return bound in lit
        return lit == bound
    if isinstance(literal_value, bool) or isinstance(bound_value, bool):
        return literal_value is bound_value
    return literal_value == bound_value


def _reject_bound_value_literals(
    old_contract: GuardContract | None,
    effective: GuardContract,
    guard_evidence: GuardEvidence,
) -> str | None:
    """All-or-nothing check: reject a new literal clause copying a runtime-bound value.

    A *new* literal predicate (present in ``effective`` but not ``old_contract``, matched by
    canonical typed identity) whose operand equals — or, for containment operators, contains —
    any frozen ``bound`` value is rejected, so a runtime binding cannot be laundered into a
    constant guard. The equivalent ``$intent.*`` clause with a matching frozen fingerprint is
    the admissible form. Returns a reason on the first match, else ``None``.
    """
    bound_values = _bound_runtime_values(guard_evidence)
    if not bound_values:
        return None
    old_predicates = (
        {_guard_predicate_identity(clause) for clause in old_contract.clauses}
        if old_contract is not None
        else set()
    )
    for clause in effective.clauses:
        expected = clause.predicate.expected
        if expected is None or expected.kind != "literal":
            continue
        if _guard_predicate_identity(clause) in old_predicates:
            continue  # already-admitted literal, not newly introduced
        operator = clause.predicate.operator
        for bound_value in bound_values:
            if _literal_copies_bound_value(expected.value, operator, bound_value):
                return (
                    "new literal clause reproduces a runtime-bound intent value; "
                    "use an $intent.* clause instead"
                )
    return None


def _guard_evidence_incomplete(guard_evidence: GuardEvidence) -> bool:
    """A cited source screen id failed to resolve to an observation.

    Admission must reason over the complete cited source-screen surface; a silently dropped
    screen would let a guard be admitted over a subset, so incomplete evidence is
    inconclusive rather than admissible.
    """
    cited = [sid for sid in guard_evidence.source_screen_ids if sid]
    return len(guard_evidence.source_observations) < len(cited)


def _sibling_admitted_contracts(fsm: AppFSM, transition: Transition) -> list[GuardContract]:
    """Admitted guard contracts of the source state's *other* outgoing transitions."""
    contracts: list[GuardContract] = []
    for other in fsm.transitions:
        if other is transition or other.source != transition.source:
            continue
        contract = other.guard_contract
        if contract is not None and contract.admission_status is GuardAdmissionStatus.ADMITTED:
            contracts.append(contract)
    return contracts


def _apply_guard_enrichment(
    transition: Transition,
    candidate: Any,
    guard_evidence: GuardEvidence,
    sibling_admitted_contracts: Sequence[GuardContract] = (),
) -> GuardEnrichmentResult:
    """Merge a guard candidate additively (or assign initially), admit, and attach.

    Never wipes an existing contract: generation failure, an intent-slot conflict, a
    rejected admission, or a source-state slot-fingerprint conflict with an already-admitted
    sibling contract all leave the transition's current guard unchanged. Only an effective
    admitted-set change that also stays source-state coherent attaches the new contract.
    """
    result = GuardEnrichmentResult()
    old_contract = transition.guard_contract
    if is_runtime_pending_placeholder(old_contract):
        # The placeholder records only "awaiting semantic admission" -- no clauses,
        # slots, or classified kind. Enrichment treats it as the initial assignment
        # rather than a merge, so the candidate's kind is adopted and admission runs
        # over the candidate alone; a merge would pin the placeholder's UNKNOWN kind
        # forever (``_merge_guard_contract`` preserves the existing kind, and
        # admission rejects UNKNOWN outright).
        old_contract = None
    if not candidate.parsed_ok:
        result.reason = candidate.rejection_reason or "structured guard generation failed"
        return result

    if old_contract is None:
        effective: GuardContract | None = candidate.contract
    else:
        effective, conflict = _merge_guard_contract(old_contract, candidate.contract)
        if effective is None:
            result.reason = conflict
            return result

    # All-or-nothing frozen-witness check for newly introduced intent-dependent clauses.
    intent_error = _validate_new_intent_clauses(old_contract, effective, guard_evidence)
    if intent_error is not None:
        result.reason = intent_error
        return result

    # All-or-nothing check: a new literal clause must not launder a runtime-bound value.
    bound_literal_error = _reject_bound_value_literals(old_contract, effective, guard_evidence)
    if bound_literal_error is not None:
        result.reason = bound_literal_error
        return result

    admission = admit_guard_contract(effective, guard_evidence)
    result.admission_status = admission.status.value
    result.rejected_predicates = list(admission.rejected_predicates)
    if not admission.admitted:
        result.reason = admission.reason
        return result

    admitted_contract = apply_guard_admission(effective, admission)
    # Post-admission dedup by compiled expression: an inherited clause carrying a resolved
    # resource id and a fresh candidate clause carrying its alias lower to the same
    # expression, so they are the same admitted clause. (Pending clauses have empty
    # compiled_expr and are deduplicated earlier by typed predicate identity instead.)
    deduped: list[GuardClause] = []
    seen_exprs: set[tuple[str | None, str]] = set()
    for clause in admitted_contract.clauses:
        key = (clause.slot, clause.compiled_expr)
        if key in seen_exprs:
            continue
        seen_exprs.add(key)
        deduped.append(clause)
    admitted_contract.clauses = deduped

    # Never silently rewrite Γ: every existing admitted clause must reproduce its exact
    # (slot, compiled_expr) in the re-admitted effective contract.
    if old_contract is not None:
        old_admitted = {
            (clause.slot, clause.compiled_expr)
            for clause in old_contract.clauses
            if clause.compiled_expr
        }
        new_admitted = {(clause.slot, clause.compiled_expr) for clause in deduped}
        if not old_admitted <= new_admitted:
            result.reason = "enrichment would rewrite an existing admitted clause"
            return result

    # Source-state coherence backstop: the staged admitted contract must agree with the
    # source state's already-admitted sibling contracts on every same-name intent-slot
    # fingerprint. This is the exact invariant the offline batch and the runtime intent-patch
    # backstop enforce; a conflict rejects promotion (no sibling regeneration, no mutation).
    coherence_pairs: list[tuple[int, GuardContract]] = [
        (index, sibling) for index, sibling in enumerate(sibling_admitted_contracts)
    ]
    coherence_pairs.append((-1, admitted_contract))
    conflicts = check_slot_fingerprint_coherence(coherence_pairs)
    if conflicts:
        names = ", ".join(conflict.slot_name for conflict in conflicts)
        result.reason = (
            f"enrichment conflicts with an admitted sibling intent-slot fingerprint: {names}"
        )
        return result

    effective_guard = (
        " && ".join(clause.compiled_expr for clause in deduped if clause.compiled_expr)
        if deduped
        else admission.guard
    )

    old_view = _guard_semantic_view(transition.guard, old_contract)
    new_view = _guard_semantic_view(effective_guard, admitted_contract)
    result.admitted = True
    result.reason = admission.reason
    if new_view == old_view:
        # Executable Γ is unchanged, but Evidence enrichment may have widened the cited
        # semantic evidence. Persist the richer contract (identical clauses/kind/status) so
        # a later audit sees the stronger support; this is not a revision-bearing change.
        if old_contract is not None:
            old_refs = {(ref.source, ref.identifier) for ref in old_contract.semantic_evidence}
            new_refs = {(ref.source, ref.identifier) for ref in admitted_contract.semantic_evidence}
            if new_refs > old_refs:
                transition.guard_contract = admitted_contract
                transition.guard = effective_guard
                result.evidence_changed = True
        return result
    transition.guard_contract = admitted_contract
    transition.guard = effective_guard
    result.changed = True
    return result


def _promote_semantics_uncommitted(
    fsm: AppFSM,
    transition: Transition,
    raw_screens: dict[str, RawScreen],
    llm: Any,
    *,
    guard_evidence_override: GuardEvidence | None = None,
) -> PromotionResult:
    """Generate and admit an additive guard contract for one existing transition.

    Uses the supplied live evidence or builds evidence for this transition, proposes
    typed clauses, and applies deterministic admission. Existing-edge evolution supplies
    its admitted live ``guard_evidence_override``. The caller records any EFSM revision
    after this operation completes; state invariants use their separate offline pipeline.
    """
    result = PromotionResult()
    transition_index = fsm.transitions.index(transition)

    guard_evidence = guard_evidence_override or build_guard_evidence_for_transition(
        fsm,
        transition,
        transition_index,
        raw_screens,
    )
    # Incomplete cited source evidence is inconclusive: skip generation/attach entirely so
    # a guard is never admitted over a subset of the cited source screens.
    if _guard_evidence_incomplete(guard_evidence):
        result.guard_incomplete = True
        result.guard_admission_reason = (
            "cited source-screen evidence is incomplete; guard promotion is inconclusive"
        )
    else:
        guard_redactor = build_prompt_redactor(fsm, [guard_evidence])
        guard_candidate = generate_llm_guard_candidate(
            guard_evidence,
            llm,
            redactor=guard_redactor,
            # A runtime placeholder is "no analysis yet", not an existing contract:
            # prompting it as existing would anchor the LLM on an empty UNKNOWN kind.
            existing_contract=(
                None
                if is_runtime_pending_placeholder(transition.guard_contract)
                else transition.guard_contract
            ),
        )
        result.guard_candidate_parsed_ok = guard_candidate.parsed_ok
        result.guard_candidate_rejection_reason = guard_candidate.rejection_reason

        # Additive Contract enrichment: an existing admitted contract is preserved and only
        # extended; a fresh transition with empty metadata gets a direct initial assignment.
        # Generation failure / slot conflict / rejected admission never wipe an existing guard.
        enrichment = _apply_guard_enrichment(
            transition,
            guard_candidate,
            guard_evidence,
            sibling_admitted_contracts=_sibling_admitted_contracts(fsm, transition),
        )
        result.guard_admission_status = enrichment.admission_status
        result.guard_admission_rejected_predicates = list(enrichment.rejected_predicates)
        result.guard_admitted = enrichment.admitted
        result.guard_admission_reason = enrichment.reason
        result.guard_changed = enrichment.changed
        result.guard_evidence_changed = enrichment.evidence_changed

    return result

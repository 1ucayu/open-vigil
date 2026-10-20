"""Symbolic transition checking over the app EFSM.

Pure symbolic checks against the FSM graph:
- Transition validity: is the proposed action legal from the current state?
- Reachability: can we still reach the goal state? O(V+E)

Trust is binary and structural, and transition graph membership is the whole of it.
There is no second trust modifier and no trust gate here. An edge is either
supported by accepted trace evidence and present, or it is absent; neither the
observation scope it came from nor its provenance downgrades it once it is in
the transition graph. The structural check is origin-agnostic: an edge created by runtime
evolution resolves here exactly like an offline one. What keeps a fresh runtime
edge from ALLOWing is not its origin but its admission state -- it is created
with a PENDING placeholder contract, and the origin-agnostic present-but-pending
rule in ``DecisionEngine`` answers UNCERTAIN until admission discharges it.

Semantic admission still gates: a present-but-rejected or still-pending guard
contract is refused through ``DecisionEngine``'s clause evaluation, regardless of
origin. And an edge whose evidence never resolved it uniquely is not gated by a
flag either -- it simply shares a canonical action key with a sibling, and
``resolve_transition`` cannot say which edge the action takes.

Returns ALLOW / DENY / UNCERTAIN with zero LLM calls.

The verdict is three-valued, and the two refusals differ in what they are about:
DENY is information about the ACTION -- a conjunct of the verification rule was
concretely refuted, so the proposal is wrong relative to a model that does cover
this behavior. UNCERTAIN is information about the MODEL -- it could not decide,
and the named gap is an evolution work item. Neither authorizes: the shield
permits only what it can prove safe (``should_execute`` is true only for ALLOW),
and a DENY is still a statement of POLICY, never a claim that the refused action
is illegal. ``PROVEN_REFUTATION_REASONS`` is the single owner of the partition,
and ``VerificationOutput.proven_refutation`` reports it as a derived audit field.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field, model_validator

from vigil.models.fsm import (
    RUNTIME_EVOLUTION_SOURCE,
    AppFSM,
    Transition,
    TransitionLookupStatus,
)
from vigil.models.guard import GuardAdmissionStatus, GuardContract, GuardKind
from vigil.models.state import RawScreen
from vigil.symbolic.clause_activator import ClauseActivationResult
from vigil.symbolic.dsl_evaluator import PredicateWitness
from vigil.symbolic.state_locator import LocateResult, StateLocator

# Action types whose safety follows from their type alone: reversible,
# non-side-effecting GUI-control actions. Everything else (click / long_press /
# input_text / unclassified) needs confirmation on an under-covered screen.
#
# Single definition, kept here rather than in ``runtime.continuation`` only
# because that module imports this one — both the verifier gate below and the
# continuation policy must agree, and duplicating the set would let them drift.
AUTO_ELIGIBLE_ACTION_TYPES = frozenset(
    {"navigate_back", "navigate_home", "scroll_up", "scroll_down"}
)


class VerifyResult(StrEnum):
    """Verification decision.

    Three-valued. ALLOW is an authorization: the shield proved this action safe
    under the model. DENY is a proven refutation: a guard or invariant concretely
    evaluated false, or an observed successor concretely mismatched -- information
    about the action, whose repair belongs to the agent. UNCERTAIN is a
    conservative refusal: the model could not decide -- information about the
    model, whose repair is an evolution work item. Both refusals withhold
    authorization (``should_execute`` is true only for ALLOW), and neither claims
    the action is illegal.

    ``VerifyReason`` carries *why*; membership of the reason in
    ``PROVEN_REFUTATION_REASONS`` is the single owner of the DENY/UNCERTAIN
    partition, enforced at construction by ``VerificationOutput``.
    """

    ALLOW = "allow"
    DENY = "deny"
    UNCERTAIN = "uncertain"


class VerifyReason(StrEnum):
    """Reason for the verification decision."""

    TRANSITION_VALID = "transition_valid"
    TRANSITION_INVALID = "transition_not_in_fsm"
    ACTION_AMBIGUOUS = "action_ambiguous"
    STATE_UNKNOWN = "state_unknown"
    GOAL_UNREACHABLE = "goal_unreachable"
    INCOMPLETE_COVERAGE = "incomplete_coverage"
    STATE_SIMILAR = "state_similar_fuzzy_match"
    GUARD_FAILED = "guard_failed"
    GUARD_INCONCLUSIVE = "guard_inconclusive"
    COMMIT_INTENT_UNBOUND = "commit_intent_unbound"
    SEMANTIC_GUARD_COVERAGE_GAP = "semantic_guard_coverage_gap"
    INVARIANT_FAILED = "invariant_failed"
    INVARIANT_INCONCLUSIVE = "invariant_inconclusive"


# The reasons on which a refusal rests on a concrete refutation rather than on
# the model's inability to decide. Membership here is the single owner of the
# DENY/UNCERTAIN partition and of ``VerificationOutput.proven_refutation``.
#
# A proven refusal (DENY) is one where the model made a checkable claim and
# reality (or the screen) falsified it:
#
#   GUARD_FAILED       an activated admitted clause conjunction evaluated FALSE
#   INVARIANT_FAILED   a state invariant evaluated FALSE on the observed screen
#   TRANSITION_INVALID the post-action screen localized to a KNOWN state that is
#                      not the predicted successor -- the model predicted, and
#                      observation refuted
#
# Everything else is a CONSERVATIVE refusal (UNCERTAIN): localization, action
# resolution, selector resolution, intent binding, or predicate evaluation was
# incomplete, so the shield refused for lack of proof. The default for any reason
# added later is therefore conservative, which is the safe direction: a new
# refusal cannot claim proof it does not have simply by being forgotten here.
#
# TRANSITION_INVALID is proven ONLY because its sole producer is the post-action
# successor check in ``runtime.monitor``. It must never be emitted for a missing
# edge at pre-action time -- that case is INCOMPLETE_COVERAGE, and it is
# conservative.
PROVEN_REFUTATION_REASONS = frozenset(
    {
        VerifyReason.GUARD_FAILED,
        VerifyReason.INVARIANT_FAILED,
        VerifyReason.TRANSITION_INVALID,
    }
)


class SemanticGuardCoverageGap(StrEnum):
    """Offline semantic asset missing from an otherwise modeled transition."""

    MISSING_CONTRACT = "missing_contract"
    REJECTED_CONTRACT = "rejected_contract"
    NO_EXECUTABLE_CLAUSE = "no_executable_clause"


def semantic_guard_coverage_gap(
    contract: GuardContract | None,
) -> SemanticGuardCoverageGap | None:
    """Classify canonical Stage-4 guard coverage gaps.

    Pending admission is an in-progress/inconclusive contract, not a rejected
    coverage asset.  Empty admitted ``none``/``navigation`` contracts are valid
    semantic-analysis results; every other admitted guard kind needs at least one
    independently admitted executable clause.
    """
    if contract is None:
        return SemanticGuardCoverageGap.MISSING_CONTRACT
    if contract.admission_status is GuardAdmissionStatus.REJECTED:
        return SemanticGuardCoverageGap.REJECTED_CONTRACT
    if contract.admission_status is not GuardAdmissionStatus.ADMITTED:
        return None
    if contract.kind in {GuardKind.NONE, GuardKind.NAVIGATION}:
        return None
    if not any(
        clause.admission_status is GuardAdmissionStatus.ADMITTED
        and bool(clause.compiled_expr.strip())
        for clause in contract.clauses
    ):
        return SemanticGuardCoverageGap.NO_EXECUTABLE_CLAUSE
    return None


def classify_semantic_admission(
    contract: GuardContract | None,
) -> tuple[VerifyReason, SemanticGuardCoverageGap | None] | None:
    """Single owner of the semantic-admission promotion classification.

    Returns ``None`` when the contract carries admitted semantic coverage,
    otherwise the uncertainty reason: a typed coverage gap, or a pending
    (not-admitted, not-rejected) contract awaiting admission.
    """
    gap = semantic_guard_coverage_gap(contract)
    if gap is not None:
        return VerifyReason.SEMANTIC_GUARD_COVERAGE_GAP, gap
    if contract is not None and contract.admission_status is not GuardAdmissionStatus.ADMITTED:
        return VerifyReason.GUARD_INCONCLUSIVE, None
    return None


class VerificationOutput(BaseModel):
    """Full verification result with reasoning.

    Attributes:
        result: ALLOW, DENY, or UNCERTAIN.
        reason: Why this decision was made.
        current_state_id: The FSM state the screen was localized to.
        target_state_id: The state the proposed action would transition to.
        details: Human-readable explanation.
        proven_refutation: Derived. True exactly for a DENY (which by
            construction rests on a concrete refutation), False for UNCERTAIN
            and ALLOW. Reported, never gating.
    """

    # ``proven_refutation`` is derived, so passing it is meaningless -- and silently
    # meaningless is the wrong failure mode for a field that exists to record whether
    # the shield has proof. Forbidding extras turns an attempt to assert proof by hand
    # into an error at the construction site instead of a value that vanishes.
    model_config = ConfigDict(extra="forbid")

    result: VerifyResult
    reason: VerifyReason
    current_state_id: str | None = None
    target_state_id: str | None = None
    details: str = ""
    clause_activation: ClauseActivationResult | None = None
    semantic_guard_gap: SemanticGuardCoverageGap | None = None
    predicate_witness: tuple[PredicateWitness, ...] | None = None
    # Per-call handoff to guard evaluation; absent from serialized results and schemas.
    _resolved_transition: Transition | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _refusal_matches_reason_partition(self) -> VerificationOutput:
        """A refusal's verdict must agree with its reason's evidentiary status.

        DENY carries a reason in ``PROVEN_REFUTATION_REASONS``; UNCERTAIN carries
        one outside it. Enforced at construction so a site that picks the wrong
        refusal verdict fails loudly instead of silently moving a record across
        the proven/conservative partition -- the same no-drift property the
        derived ``proven_refutation`` field has, extended to the verdict itself.
        ALLOW is unconstrained.
        """
        if self.result is VerifyResult.DENY and self.reason not in PROVEN_REFUTATION_REASONS:
            raise ValueError(
                f"DENY requires a proven-refutation reason; {self.reason.value!r} is "
                "conservative and must be UNCERTAIN"
            )
        if self.result is VerifyResult.UNCERTAIN and self.reason in PROVEN_REFUTATION_REASONS:
            raise ValueError(
                f"UNCERTAIN cannot carry the proven-refutation reason {self.reason.value!r}; "
                "a concrete refutation must be DENY"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def proven_refutation(self) -> bool:
        """Whether this refusal is proof-backed rather than conservative.

        Derived from ``(result, reason)`` rather than set at each construction
        site, so the two can never drift apart and a refusal can never claim
        proof by accident. See ``PROVEN_REFUTATION_REASONS``. With the
        construction-time partition check this is equivalent to
        ``result is DENY``; it stays reason-derived so the audit field and the
        verdict can never disagree even if one of them changes.

        This distinction is the whole evidentiary content of a refusal: it is what
        separates prevention the shield can defend from over-refusal it cannot,
        and it is reported, never consulted as a gate.
        """
        return self.result is VerifyResult.DENY and self.reason in PROVEN_REFUTATION_REASONS


def is_conservative_refusal(output: VerificationOutput) -> bool:
    """Whether this is a refusal the model could not back with proof.

    Exactly ``result is UNCERTAIN``: the conservative refusals are the UNCERTAIN
    verdicts, and the proven ones are the DENYs. The named predicate survives the
    verdict restoration so callers keep reading as intent rather than enum
    comparison, and so both predicates stay total over every refusal.
    """
    return output.result is VerifyResult.UNCERTAIN


class FsmChecker:
    """Check structural support in the app EFSM.

    Takes a current screen and proposed action and returns a structural
    verdict using graph lookups on the pre-built EFSM.

    Args:
        fsm: The app's verified FSM.
    """

    def __init__(self, fsm: AppFSM) -> None:
        self._fsm = fsm
        self._locator = StateLocator(fsm)

    def verify_structure(
        self,
        current_screen: RawScreen,
        proposed_action: dict[str, Any],
        goal_state: str | None = None,
        stage_timings_out: dict[str, float] | None = None,
    ) -> VerificationOutput:
        """Localize the source observation and resolve its canonical transition."""
        started = time.perf_counter()
        location = self._locator.locate(current_screen)
        if stage_timings_out is not None:
            stage_timings_out["localization_ms"] = (time.perf_counter() - started) * 1000.0

        if location.result == LocateResult.UNKNOWN:
            return VerificationOutput(
                result=VerifyResult.UNCERTAIN,
                reason=VerifyReason.STATE_UNKNOWN,
                details="Current screen does not match any known FSM state",
            )

        if location.result == LocateResult.SIMILAR:
            return VerificationOutput(
                result=VerifyResult.UNCERTAIN,
                reason=VerifyReason.STATE_SIMILAR,
                current_state_id=location.state_id,
                details=(
                    f"Screen matched state {location.state_id} via the non-authoritative "
                    "functional-hash fallback, not a deterministic localization"
                ),
            )

        # Deterministic known match — proceed with structural verification.
        assert location.state_id is not None
        return self.verify_structure_by_state(location.state_id, proposed_action, goal_state)

    def verify_by_state(
        self,
        current_state_id: str,
        proposed_action: dict[str, Any],
        goal_state: str | None = None,
    ) -> VerificationOutput:
        """Verify when the current FSM state is already known.

        Args:
            current_state_id: The current FSM state ID.
            proposed_action: Action dict (e.g., {"type": "click"}).
            goal_state: Optional goal state ID for reachability checking.

        Returns:
            VerificationOutput with decision and reasoning.
        """
        return self.verify_structure_by_state(
            current_state_id,
            proposed_action,
            goal_state,
        )

    def verify_structure_by_state(
        self,
        current_state_id: str,
        proposed_action: dict[str, Any],
        goal_state: str | None = None,
    ) -> VerificationOutput:
        """Check canonical transition membership and optional goal reachability."""
        # 1. Transition validity and action identity resolution
        lookup = self._fsm.resolve_transition(current_state_id, proposed_action)
        if lookup.status is TransitionLookupStatus.UNCERTAIN:
            return VerificationOutput(
                result=VerifyResult.UNCERTAIN,
                reason=VerifyReason.ACTION_AMBIGUOUS,
                current_state_id=current_state_id,
                details=lookup.details,
            )
        if lookup.status is not TransitionLookupStatus.MATCH:
            # Missing edge from a known state. The shield refuses -- and the refusal
            # is CONSERVATIVE, not evidential. Missing evidence never refutes.
            #
            # The graph records observed transitions, not every behavior the app
            # permits. Refuting a proposal on a missing edge would require a
            # closed-world claim at the granularity of a (state, action) PAIR: that
            # this action, at this state, was tried and found impossible. Exploration
            # cannot establish that. A state having been observed, and some *other*
            # action having been tried there, licenses nothing about the action
            # actually proposed -- so the closed-world premise fails on every state,
            # not just under-explored ones.
            #
            # So the UNCERTAIN below says "the model cannot prove this action safe,
            # therefore the shield does not authorize it". It does NOT say the action
            # is illegal, and nothing downstream may read it that way: this is
            # information about the model, not about the action, which is why
            # INCOMPLETE_COVERAGE is outside PROVEN_REFUTATION_REASONS and the named
            # gap is precisely an evolution work item (coverage expansion).
            source_state = self._fsm.states.get(current_state_id)
            scope = (
                "runtime-evolved state"
                if source_state is not None
                and source_state.evidence.construction_source == RUNTIME_EVOLUTION_SOURCE
                else "state"
            )
            return VerificationOutput(
                result=VerifyResult.UNCERTAIN,
                reason=VerifyReason.INCOMPLETE_COVERAGE,
                current_state_id=current_state_id,
                details=(
                    f"Action {proposed_action.get('type')} is not a modeled transition "
                    f"from {scope} {current_state_id}; the graph records observed behaviour, "
                    "so its silence on this action is not a refutation"
                ),
            )

        # 2. Get target state
        target_id = lookup.target_state_id

        # 3. Goal reachability
        #
        # Same principle as the missing edge above: unreachability is computed over
        # the transition graph, which records observed behavior rather than every
        # behavior the app permits. "No path in the model" is not "no path in the app",
        # so this cannot refute either -- the refusal below is conservative, and
        # ``proven_refutation`` is False on it.
        #
        if (
            goal_state is not None
            and target_id is not None
            and not self._fsm.is_reachable(target_id, goal_state)
        ):
            return VerificationOutput(
                result=VerifyResult.UNCERTAIN,
                reason=VerifyReason.GOAL_UNREACHABLE,
                current_state_id=current_state_id,
                target_state_id=target_id,
                details=(
                    f"Goal state {goal_state} is not reachable from target state "
                    f"{target_id} in the model; the model's silence is not a refutation"
                ),
            )

        result = VerificationOutput(
            result=VerifyResult.ALLOW,
            reason=VerifyReason.TRANSITION_VALID,
            current_state_id=current_state_id,
            target_state_id=target_id,
            details=f"Valid transition from {current_state_id} to {target_id}",
        )
        result._resolved_transition = lookup.transition
        return result

"""Deterministic execution-time shield for proposed GUI actions.

Combines structural transition checks with admitted guard evaluation and returns
ALLOW / DENY / UNCERTAIN without invoking an LLM. Refusals provide evidence for
the agent and for a separate construction or evolution round.

Only the guard and invariant checks below can produce a proven DENY; every other
refusal here is UNCERTAIN -- the engine could not decide, and refusing on that
basis is a policy choice rather than a claim about the action.
"""

from __future__ import annotations

import time
from typing import Any

from vigil.models.fsm import AppFSM, Transition
from vigil.models.guard import GuardKind
from vigil.models.intent_binding import IntentBindingSnapshot
from vigil.models.state import RawScreen
from vigil.symbolic.clause_activator import (
    ClauseActivationResult,
    ClauseActivator,
    ClauseDisposition,
)
from vigil.symbolic.dsl_evaluator import (
    DSLEvaluator,
    GuardResult,
    GuardStatus,
    IntentContext,
    ScreenContext,
)
from vigil.symbolic.fsm_checker import (
    FsmChecker,
    SemanticGuardCoverageGap,
    VerificationOutput,
    VerifyReason,
    VerifyResult,
    classify_semantic_admission,
)
from vigil.symbolic.invariant_checker import InvariantChecker
from vigil.symbolic.screen_context import screen_context_from_observation


class DecisionEngine:
    """Check transition structure and admitted transition guardrails.

    State localization and transition resolution establish structural support.
    Activated guard clauses then evaluate against the source observation,
    proposed action, and frozen intent bindings.

    Args:
        fsm: The app's verified FSM.
    """

    def __init__(
        self,
        fsm: AppFSM,
    ) -> None:
        self._fsm = fsm
        self._checker = FsmChecker(fsm)
        self._evaluator = DSLEvaluator()
        self._clause_activator = ClauseActivator()
        self._invariant_checker = InvariantChecker(fsm, evaluator=self._evaluator)

    def verify(
        self,
        current_screen: RawScreen,
        proposed_action: dict[str, Any],
        goal_state: str | None = None,
        binding_snapshot: IntentBindingSnapshot | None = None,
        stage_timings_out: dict[str, float] | None = None,
    ) -> VerificationOutput:
        """The master VERIFY function with full screen input.

        Args:
            current_screen: The current device screen.
            proposed_action: Action dict (e.g., {"type": "click", "target": "e_001"}).
            goal_state: Optional goal state for reachability checking.
            binding_snapshot: Frozen status-aware bindings supplied by the mobile agent.
            stage_timings_out: Optional dict filled with additive per-stage
                wall-clock durations (ms). Observational only; never a verdict input.

        Returns:
            VerificationOutput with decision and reasoning.
        """

        def _stamp(name: str, t0: float) -> None:
            if stage_timings_out is not None:
                stage_timings_out[name] = (time.perf_counter() - t0) * 1000.0

        # Resolve the transition first. Graph membership is the whole of trust,
        # so once an edge resolves the only remaining gate is its admitted semantic
        # clauses -- a proven guard FALSE is a decisive DENY.
        _t0 = time.perf_counter()
        structural_result = self._checker.verify_structure(
            current_screen,
            proposed_action,
            goal_state,
            stage_timings_out=stage_timings_out,
        )
        _stamp("transition_resolution_ms", _t0)

        if structural_result.result is not VerifyResult.ALLOW:
            return structural_result

        transition = structural_result._resolved_transition

        _t0 = time.perf_counter()
        bound_target = proposed_action.get("target")
        screen_ctx = self._build_screen_context(
            current_screen,
            focus_element_id=bound_target if isinstance(bound_target, str) else None,
        )
        action_ctx = self._build_action_context(proposed_action, current_screen)
        _stamp("guard_context_build_ms", _t0)
        _t0 = time.perf_counter()
        result = self._verify_transition_guard(
            transition,
            structural_result,
            binding_snapshot=binding_snapshot,
            screen_ctx=screen_ctx,
            action_ctx=action_ctx,
        )
        _stamp("guard_clause_eval_ms", _t0)
        return result

    def verify_by_state(
        self,
        current_state_id: str,
        proposed_action: dict[str, Any],
        screen_ctx: ScreenContext | None = None,
        action_ctx: dict[str, Any] | None = None,
        goal_state: str | None = None,
        binding_snapshot: IntentBindingSnapshot | None = None,
    ) -> VerificationOutput:
        """Verify when state is already known (skip localization).

        Args:
            current_state_id: The current FSM state ID.
            proposed_action: Action dict.
            screen_ctx: Screen context for guard evaluation.
            action_ctx: Action metadata for action_pred evaluation.
            goal_state: Optional goal state for reachability checking.
            binding_snapshot: Frozen status-aware bindings supplied by the mobile agent.

        Returns:
            VerificationOutput with decision and reasoning.
        """
        structural_result = self._checker.verify_structure_by_state(
            current_state_id,
            proposed_action,
            goal_state,
        )

        if structural_result.result is not VerifyResult.ALLOW:
            return structural_result

        transition = structural_result._resolved_transition

        if screen_ctx is None:
            screen_ctx = ScreenContext()
        if action_ctx is None:
            # No RawScreen here — build a minimal action context straight from the
            # proposed action so action(input_text)/action(target_text) guards evaluate.
            action_ctx = self._build_minimal_action_context(proposed_action)

        result = self._verify_transition_guard(
            transition,
            structural_result,
            binding_snapshot=binding_snapshot,
            screen_ctx=screen_ctx,
            action_ctx=action_ctx,
        )
        return result

    def check_state_invariants(
        self,
        target_state_id: str,
        observed_target_screen: RawScreen,
    ) -> VerificationOutput:
        """Check target-state invariants against an observed target screen.

        This method is the public hook for enforcing state invariants once the caller
        supplies the screen currently localized to ``target_state_id``.
        """
        if target_state_id not in self._fsm.states:
            return VerificationOutput(
                result=VerifyResult.UNCERTAIN,
                reason=VerifyReason.STATE_UNKNOWN,
                target_state_id=target_state_id,
                details=f"Target state {target_state_id} is not in the FSM",
            )

        screen_ctx = self._build_screen_context(observed_target_screen)
        invariant_result = self._check_invariants(
            target_state_id=target_state_id,
            screen_ctx=screen_ctx,
        )
        if invariant_result is not None:
            return invariant_result
        return VerificationOutput(
            result=VerifyResult.ALLOW,
            reason=VerifyReason.TRANSITION_VALID,
            target_state_id=target_state_id,
            details=f"All invariants passed for {target_state_id}",
        )

    def _verify_transition_guard(
        self,
        transition: Transition,
        structural_result: VerificationOutput,
        *,
        binding_snapshot: IntentBindingSnapshot | None,
        screen_ctx: ScreenContext,
        action_ctx: dict[str, Any],
    ) -> VerificationOutput:
        """Run the admitted-clause activation and DSL evaluation stages."""
        contract = transition.guard_contract
        # An evidence-backed transition without a semantic contract uses the
        # neutral guard Gamma(s, sigma) = true. Preserve the structural result.
        # This rule is independent of the edge's origin. Newly witnessed edges
        # carry a pending contract, so they remain UNCERTAIN until admission.
        #
        # Everything below is origin-agnostic and still gates: a present contract
        # that is rejected routes to a coverage gap, and one still pending routes to
        # inconclusive -- both UNCERTAIN. Only the absence of a contract reaches the
        # neutral guard.
        if contract is None:
            return structural_result

        classification = classify_semantic_admission(contract)
        if classification is not None:
            _, coverage_gap = classification
            if coverage_gap is not None:
                return self._semantic_guard_coverage_gap(structural_result, coverage_gap)
            return self._guard_inconclusive(
                structural_result,
                f"guard contract is not admitted ({contract.admission_status.value})",
            )

        clauses = contract.clauses
        if not clauses:
            return structural_result

        snapshot = binding_snapshot or IntentBindingSnapshot()
        activation = self._clause_activator.activate(contract, snapshot)
        evaluation_intent = IntentContext(
            variables=dict(activation.bound_variables),
        )

        # Scale-incoherent numeric bindings poison every clause that reads the
        # affected source scalar: one comparison would be evaluated in the wrong
        # scale, so a FALSE there is a scale artifact (not a proven violation) and
        # a TRUE there is vacuous (not semantic coverage). Withhold BEFORE
        # evaluation — this must never reach the decisive-DENY or ALLOW paths.
        if activation.scale_incoherence:
            return self._guard_inconclusive(
                structural_result,
                "numeric scale incoherence: " + "; ".join(activation.scale_incoherence),
                activation=activation,
            )

        # Evaluate every active clause as one temporary conjunction. A proven
        # FALSE remains decisive even when a different clause is inconclusive.
        expression = activation.expression
        if expression is not None:
            guard_result = self._evaluator.evaluate(
                expression,
                intent_ctx=evaluation_intent,
                screen_ctx=screen_ctx,
                action_ctx=action_ctx,
                record_witness=True,
            )
            routed = self._route_guard_result(
                guard_result, structural_result, activation=activation
            )
            if routed is not None:
                return routed
        if activation.has_inconclusive:
            reasons = ", ".join(
                clause.reason
                for clause in activation.clauses
                if clause.disposition is ClauseDisposition.INCONCLUSIVE
            )
            return self._guard_inconclusive(
                structural_result,
                f"clause activation inconclusive: {reasons}",
                activation=activation,
            )
        # Commit gate: a commit/safety contract whose semantic protection is
        # entirely intent-bound must not release the action while every
        # intent-bound clause is skipped (no slot bound). A proven FALSE above
        # stays DENY; one active intent-bound clause is enough to proceed to
        # the normal ALLOW path; literal-only commit contracts pass unchanged.
        if contract.kind in {GuardKind.CONFIRM_COMMIT, GuardKind.SAFETY_CHECK}:
            intent_bound = [clause for clause in activation.clauses if clause.slot is not None]
            if intent_bound and not any(
                clause.disposition is ClauseDisposition.ACTIVE for clause in intent_bound
            ):
                skipped_slots = sorted(
                    {clause.slot for clause in intent_bound if clause.slot is not None}
                )
                return VerificationOutput(
                    result=VerifyResult.UNCERTAIN,
                    reason=VerifyReason.COMMIT_INTENT_UNBOUND,
                    current_state_id=structural_result.current_state_id,
                    target_state_id=structural_result.target_state_id,
                    details=(
                        "commit/safety guard has no active intent-bound clause: "
                        "bind at least one of " + ", ".join(skipped_slots)
                    ),
                    clause_activation=activation,
                )
        return structural_result.model_copy(update={"clause_activation": activation})

    @staticmethod
    def _guard_inconclusive(
        structural_result: VerificationOutput,
        details: str,
        activation: ClauseActivationResult | None = None,
    ) -> VerificationOutput:
        return VerificationOutput(
            result=VerifyResult.UNCERTAIN,
            reason=VerifyReason.GUARD_INCONCLUSIVE,
            current_state_id=structural_result.current_state_id,
            target_state_id=structural_result.target_state_id,
            details=details,
            clause_activation=activation,
        )

    @staticmethod
    def _semantic_guard_coverage_gap(
        structural_result: VerificationOutput,
        gap: SemanticGuardCoverageGap,
    ) -> VerificationOutput:
        return VerificationOutput(
            result=VerifyResult.UNCERTAIN,
            reason=VerifyReason.SEMANTIC_GUARD_COVERAGE_GAP,
            current_state_id=structural_result.current_state_id,
            target_state_id=structural_result.target_state_id,
            details=f"Offline semantic guard coverage gap: {gap.value}",
            semantic_guard_gap=gap,
        )

    def _route_guard_result(
        self,
        guard_result: GuardResult,
        structural_result: VerificationOutput,
        activation: ClauseActivationResult | None = None,
    ) -> VerificationOutput | None:
        """Map a three-valued GuardResult to a VerificationOutput.

        Returns None when the guard is TRUE (ALLOW preserved upstream),
        a proven DENY when FALSE, an UNCERTAIN when UNKNOWN.
        """
        witness = guard_result.witness
        if guard_result.status is GuardStatus.TRUE:
            return None
        if guard_result.status is GuardStatus.UNKNOWN:
            return VerificationOutput(
                result=VerifyResult.UNCERTAIN,
                reason=VerifyReason.GUARD_INCONCLUSIVE,
                current_state_id=structural_result.current_state_id,
                target_state_id=structural_result.target_state_id,
                details=(
                    f"Guard inconclusive: {guard_result.guard_expression}"
                    f" — {guard_result.failure_reason}"
                ),
                clause_activation=activation,
                predicate_witness=witness,
            )
        return VerificationOutput(
            result=VerifyResult.DENY,
            reason=VerifyReason.GUARD_FAILED,
            current_state_id=structural_result.current_state_id,
            target_state_id=structural_result.target_state_id,
            details=(
                f"Guard failed: {guard_result.guard_expression} → {guard_result.failure_reason}"
            ),
            clause_activation=activation,
            predicate_witness=witness,
        )

    def _check_invariants(
        self,
        target_state_id: str,
        screen_ctx: ScreenContext,
    ) -> VerificationOutput | None:
        """Enforce invariant map I on an observed state.

        Returns None when all invariants hold (TRUE) or none exist, DENY on any FALSE,
        UNCERTAIN on any UNKNOWN (with no FALSE).
        """
        inv = self._invariant_checker.check_state(target_state_id, screen_ctx)
        if inv.failed > 0:
            first_expr, first_reason = inv.failed_invariants[0]
            return VerificationOutput(
                result=VerifyResult.DENY,
                reason=VerifyReason.INVARIANT_FAILED,
                target_state_id=target_state_id,
                details=f"Invariant failed on {target_state_id}: {first_expr} — {first_reason}",
            )
        if inv.unknown > 0:
            first_expr, first_reason = inv.unknown_invariants[0]
            return VerificationOutput(
                result=VerifyResult.UNCERTAIN,
                reason=VerifyReason.INVARIANT_INCONCLUSIVE,
                target_state_id=target_state_id,
                details=f"Invariant inconclusive on {target_state_id}:"
                f" {first_expr} — {first_reason}",
            )
        return None

    @staticmethod
    def _build_screen_context(
        screen: RawScreen, focus_element_id: str | None = None
    ) -> ScreenContext:
        """Convert one runtime screen through the canonical DSL context builder.

        ``focus_element_id`` is the live element the action bound to. Passing it
        lets a guard over a repeated row read *that* row rather than collapsing to
        an ambiguous alias; it is omitted for post-action invariant checks, which
        judge the whole screen and not one row.
        """
        return screen_context_from_observation(screen, focus_element_id=focus_element_id)

    @staticmethod
    def _build_action_context(
        proposed_action: dict[str, Any],
        current_screen: RawScreen,
    ) -> dict[str, Any]:
        """Build action context for action_pred evaluation.

        Extracts the target element's text and resource_id from current_screen
        using the action's target element_id. Also exposes ``input_text`` (the typed
        value of an input action) so ``action(input_text)`` guards can evaluate.
        """
        ctx: dict[str, Any] = {
            "action_type": proposed_action.get("type", ""),
            "input_text": proposed_action.get("input_text", proposed_action.get("text", "")),
        }
        target_id = proposed_action.get("target")
        if target_id and current_screen:
            for e in current_screen.elements:
                if e.element_id == target_id:
                    ctx["target_text"] = e.text or ""
                    ctx["target_resource_id"] = e.resource_id or ""
                    ctx["target_content_desc"] = e.content_description or ""
                    break
        return ctx

    @staticmethod
    def _build_minimal_action_context(
        proposed_action: dict[str, Any],
    ) -> dict[str, Any]:
        """Action context built solely from the proposed action (no RawScreen).

        Used by ``verify_by_state`` when no ``action_ctx`` is supplied so that
        ``action(...)`` guards — notably ``action(input_text)`` and
        ``action(target_text)`` — can still evaluate.
        """
        ctx: dict[str, Any] = {
            "action_type": proposed_action.get("type", ""),
            "input_text": proposed_action.get("input_text", proposed_action.get("text", "")),
        }
        if "target_text" in proposed_action:
            ctx["target_text"] = proposed_action.get("target_text") or ""
        resource_id = proposed_action.get("target_resource_id") or proposed_action.get(
            "resource_id"
        )
        if resource_id is not None:
            ctx["target_resource_id"] = resource_id or ""
        if "target_content_desc" in proposed_action:
            ctx["target_content_desc"] = proposed_action.get("target_content_desc") or ""
        return ctx

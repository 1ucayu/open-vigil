"""RuntimeMonitor: the generic online verification loop.

Orchestrates one deterministic symbolic verification pass over a proposed GUI
action:

    freeze bindings -> bind action -> localize and verify

with a separate public :meth:`RuntimeMonitor.verify_post_action` for the
post-action target-match + invariant check once the caller has observed the
post-action screen.

Design invariants (paper-aligned):

- Common path is symbolic and deterministic. No runtime LLM decision path.
- The verdict is three-valued: the shield authorizes with ALLOW, refutes with
  DENY, and refuses conservatively with UNCERTAIN. Incomplete evidence (unknown
  state, ambiguous/unbound action, an inconclusive clause activation, or a
  post-action screen that cannot be localized) is UNCERTAIN -- the shield could
  not prove the step safe, which is not a claim that the step was unsafe. Only a
  *proven* violation (wrong known target state, or a failed invariant) is DENY
  and carries ``proven_refutation``.
- Action identity is bound through the existing ``resolve_transition`` /
  ``canonical_action_key`` semantics. No new action alphabet.

Guard admission, clause activation, and DSL evaluation live together in
``DecisionEngine`` so an unadmitted guard can never produce a runtime DENY.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from vigil.models.fsm import AppFSM, TransitionLookupStatus
from vigil.models.intent_binding import IntentBindingSnapshot
from vigil.models.state import RawScreen
from vigil.runtime.action_binder import ActionBinder, BindingStatus, ProposedAction
from vigil.runtime.session import RuntimeSession
from vigil.symbolic.decision_engine import DecisionEngine
from vigil.symbolic.fsm_checker import VerificationOutput, VerifyReason, VerifyResult
from vigil.symbolic.state_locator import LocateResult, StateLocator, is_known_location


class RuntimeVerificationResult(BaseModel):
    """Structured outcome of one runtime verification pass.

    Attributes:
        pre_result: Pre-action verdict (after the guard-required policy).
        final_result: Authoritative verdict — ``post_result`` when a post-action
            check ran, otherwise ``pre_result``.
        bound_action: The action dict fed to the verifier (post-binding).
        binding_status: How the proposed action bound to the live screen.
        binding_reason: Diagnostic for the binding outcome.
        pre_state_id: State the pre-action screen localized to (if any).
        expected_target_state_id: Target the resolved transition points at.
        observed_target_state_id: State the post-action screen localized to.
        post_result: Post-action target/invariant verdict (None if not run).
        binding_snapshot: Exact status-aware snapshot supplied by the mobile
            agent. Omitted slots remain missing rather than becoming absent.
    """

    pre_result: VerificationOutput
    final_result: VerificationOutput
    bound_action: dict[str, Any]
    binding_status: BindingStatus
    binding_reason: str = ""
    pre_state_id: str | None = None
    expected_target_state_id: str | None = None
    observed_target_state_id: str | None = None
    post_result: VerificationOutput | None = None
    binding_snapshot: IntentBindingSnapshot | None = None


class RuntimeMonitor:
    """Agent-agnostic online verifier composed from the existing symbolic layer.

    Args:
        fsm: The app's verified FSM.
    """

    def __init__(self, fsm: AppFSM) -> None:
        self._fsm = fsm
        self._engine = DecisionEngine(fsm)
        self._locator = StateLocator(fsm)
        self._binder = ActionBinder()

    def verify_action(
        self,
        screen: RawScreen,
        proposed_action: dict[str, Any] | ProposedAction,
        session: RuntimeSession | None = None,
        binding_snapshot: IntentBindingSnapshot | None = None,
        goal_state: str | None = None,
        stage_timings_out: dict[str, float] | None = None,
    ) -> RuntimeVerificationResult:
        """Run one pre-action verification pass.

        ``stage_timings_out``, when supplied, is filled with additive per-stage
        wall-clock durations (ms). Purely observational; never affects verdicts.
        """
        # A session snapshot is frozen for the episode. Stateless callers pass the
        # equally frozen snapshot explicitly for this verification.
        resolved_snapshot = (
            session.resolve_binding_snapshot(binding_snapshot)
            if session is not None
            else binding_snapshot
        )

        # Bind the proposed action to a live element. Any non-MATCH is
        #    refused, returned before DecisionEngine can produce an ALLOW.
        _stage_t0 = time.perf_counter()
        binding = self._binder.bind(proposed_action, screen)
        if stage_timings_out is not None:
            stage_timings_out["action_binding_ms"] = (time.perf_counter() - _stage_t0) * 1000.0
        bound_action = binding.bound_action
        if binding.status is not BindingStatus.MATCH:
            # Binding failures still need source-state context for feedback.
            # Bound actions are localized once by the decision engine below.
            _stage_t0 = time.perf_counter()
            pre_state_id = self._locator.locate(screen).state_id
            if stage_timings_out is not None:
                stage_timings_out["localization_ms"] = (time.perf_counter() - _stage_t0) * 1000.0
            pre = VerificationOutput(
                result=VerifyResult.UNCERTAIN,
                reason=VerifyReason.ACTION_AMBIGUOUS,
                current_state_id=pre_state_id,
                details=f"action binding {binding.status.value}: {binding.reason}",
            )
            return RuntimeVerificationResult(
                pre_result=pre,
                final_result=pre,
                bound_action=bound_action,
                binding_status=binding.status,
                binding_reason=binding.reason,
                pre_state_id=pre_state_id,
                binding_snapshot=resolved_snapshot,
            )

        # Pre-action verification owns localization, structure, and guard evaluation.
        pre_result = self._engine.verify(
            screen,
            bound_action,
            goal_state=goal_state,
            binding_snapshot=resolved_snapshot,
            stage_timings_out=stage_timings_out,
        )

        pre_state_id = pre_result.current_state_id
        expected_target = pre_result.target_state_id
        # A non-authoritative similar-state match can still carry a candidate
        # successor for diagnostics. Authoritative results already include it.
        if pre_result.reason is VerifyReason.STATE_SIMILAR and pre_state_id is not None:
            lookup = self._fsm.resolve_transition(pre_state_id, bound_action)
            if lookup.status is TransitionLookupStatus.MATCH:
                expected_target = lookup.target_state_id
        return RuntimeVerificationResult(
            pre_result=pre_result,
            final_result=pre_result,
            bound_action=bound_action,
            binding_status=binding.status,
            binding_reason=binding.reason,
            pre_state_id=pre_state_id,
            expected_target_state_id=expected_target,
            binding_snapshot=resolved_snapshot,
        )

    # -- helpers ---------------------------------------------------------------

    def verify_post_action(
        self,
        post_screen: RawScreen,
        expected_target: str | None,
        stage_timings_out: dict[str, float] | None = None,
    ) -> tuple[VerificationOutput, str | None]:
        """Localize an observed post-action screen and enforce target invariants."""
        _stage_t0 = time.perf_counter()
        post_loc = self._locator.locate(post_screen)
        if stage_timings_out is not None:
            stage_timings_out["localization_ms"] = (time.perf_counter() - _stage_t0) * 1000.0
        observed = post_loc.state_id

        if is_known_location(post_loc.result):
            if expected_target is not None and observed != expected_target:
                # Landed in a *known* wrong state — a proven target mismatch.
                return (
                    VerificationOutput(
                        result=VerifyResult.DENY,
                        reason=VerifyReason.TRANSITION_INVALID,
                        current_state_id=expected_target,
                        target_state_id=observed,
                        details=(f"post-action state {observed} != expected {expected_target}"),
                    ),
                    observed,
                )
            # Observed matches the expected target (or none was known): enforce
            # the target state's invariants against the observed screen.
            target_for_invariants = expected_target if expected_target is not None else observed
            if target_for_invariants is None:
                return (
                    VerificationOutput(
                        result=VerifyResult.UNCERTAIN,
                        reason=VerifyReason.STATE_UNKNOWN,
                        details="post-action target state is unknown",
                    ),
                    observed,
                )
            _stage_t0 = time.perf_counter()
            invariant_output = self._engine.check_state_invariants(
                target_for_invariants, post_screen
            )
            if stage_timings_out is not None:
                stage_timings_out["invariant_check_ms"] = (time.perf_counter() - _stage_t0) * 1000.0
            return (invariant_output, observed)

        # SIMILAR / UNKNOWN — cannot prove where we landed. Refused, but
        # conservatively: an unlocalizable landing is not evidence of a wrong one.
        reason = (
            VerifyReason.STATE_SIMILAR
            if post_loc.result is LocateResult.SIMILAR
            else VerifyReason.STATE_UNKNOWN
        )
        return (
            VerificationOutput(
                result=VerifyResult.UNCERTAIN,
                reason=reason,
                current_state_id=observed,
                target_state_id=expected_target,
                details=(
                    f"post-action localization {post_loc.result.value};"
                    f" cannot confirm target {expected_target}"
                ),
            ),
            observed,
        )

"""State invariant checking (Daikon-style).

Verifies that state-level invariants hold for the current screen.
Called when a screen has been localized to an abstract state.
Invariants are mined during offline FSM construction (Stage 2.5)
and stored in AbstractState.invariant_specs.

Three-valued (TRUE / FALSE / UNKNOWN). DecisionEngine routes:
  any FALSE -> DENY (invariant violation proven);
  any UNKNOWN with no FALSE -> UNCERTAIN (cannot prove safety).
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from vigil.models.fsm import AppFSM
from vigil.symbolic.dsl_evaluator import DSLEvaluator, GuardStatus, ScreenContext


class InvariantCheckResult(BaseModel):
    """Result of checking all invariants for a state."""

    failed: int = 0
    unknown: int = 0
    failed_invariants: list[tuple[str, str]] = Field(default_factory=list)
    unknown_invariants: list[tuple[str, str]] = Field(default_factory=list)


class InvariantChecker:
    """Checks state invariants against runtime screen state.

    Uses DSLEvaluator to evaluate each invariant expression from
    AbstractState.invariant_specs against the current ScreenContext.
    """

    def __init__(self, fsm: AppFSM, evaluator: DSLEvaluator | None = None) -> None:
        self._fsm = fsm
        self._evaluator = evaluator or DSLEvaluator()

    def check_state(
        self,
        state_id: str,
        screen_ctx: ScreenContext,
    ) -> InvariantCheckResult:
        """Check all invariants for a state against current screen.

        An unknown ``state_id`` is incomplete evidence and therefore an
        inconclusive result (never an implicit pass).
        """
        state = self._fsm.states.get(state_id)
        if state is None:
            reason = f"State {state_id} not found in FSM"
            logger.warning(reason)
            return InvariantCheckResult(unknown=1, unknown_invariants=[("", reason)])

        failed_list: list[tuple[str, str]] = []
        unknown_list: list[tuple[str, str]] = []

        for spec in state.invariant_specs:
            inv_expr = spec.expr
            result = self._evaluator.evaluate(inv_expr, screen_ctx=screen_ctx)
            if result.status is GuardStatus.TRUE:
                continue
            if result.status is GuardStatus.UNKNOWN:
                reason = result.failure_reason or f"Invariant inconclusive: {inv_expr}"
                unknown_list.append((inv_expr, reason))
                logger.debug(f"Invariant unknown for {state_id}: {inv_expr} — {reason}")
            else:
                reason = result.failure_reason or f"Invariant evaluated to False: {inv_expr}"
                failed_list.append((inv_expr, reason))
                logger.debug(f"Invariant failed for {state_id}: {inv_expr} — {reason}")

        return InvariantCheckResult(
            failed=len(failed_list),
            unknown=len(unknown_list),
            failed_invariants=failed_list,
            unknown_invariants=unknown_list,
        )

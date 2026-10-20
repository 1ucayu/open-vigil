"""Planner-time verification for candidate GUI actions.

This module exposes the verifier inside a planner loop. It checks candidate
next actions from the planner's current abstract state and returns an action
mask plus per-candidate reasons. It does not simulate a full precomputed plan
and does not inspect a live screen; execution-time verification remains the
responsibility of :mod:`vigil.symbolic.decision_engine`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from vigil.models.fsm import AppFSM
from vigil.symbolic.fsm_checker import FsmChecker, VerificationOutput, VerifyResult


class PlannerCandidateVerification(BaseModel):
    """Per-candidate planner-time verification result."""

    index: int
    action: dict[str, Any]
    result: VerificationOutput
    selectable: bool


class PlannerVerification(BaseModel):
    """Action-mask style guidance returned to a planner."""

    overall_result: VerifyResult
    candidate_results: list[PlannerCandidateVerification]
    action_mask: list[bool]


class PlannerVerifier:
    """Verify candidate actions before the planner commits to one.

    Plan-time guidance checks structural support: it runs
    ``FsmChecker.verify_by_state`` (canonical transition membership) and never evaluates
    guards or invariants. Guard/invariant clauses are surfaced elsewhere as *slice
    information* for the planner, but evaluating a guard requires the frozen binding
    snapshot and the live screen, which exist only in the execution-time shield
    (:mod:`vigil.symbolic.decision_engine`). The action mask is therefore advisory: a
    transition carrying an admitted guard that will certainly ``DENY`` at execution can
    still be marked ``selectable=True`` here.
    """

    def __init__(self, fsm: AppFSM) -> None:
        self._checker = FsmChecker(fsm)

    def verify_candidates(
        self,
        current_state_id: str,
        candidates: list[dict[str, Any]],
        goal_state: str | None = None,
    ) -> PlannerVerification:
        """Verify candidate next actions and return a planner action mask.

        A candidate is selectable only when the structural verifier returns
        ``ALLOW``. ``UNCERTAIN`` candidates are preserved for repair/fallback,
        but their mask entry is ``False``.
        """
        if not candidates:
            # An empty candidate set proves nothing about any action: the model
            # was never consulted, so the aggregate is conservative.
            return PlannerVerification(
                overall_result=VerifyResult.UNCERTAIN,
                candidate_results=[],
                action_mask=[],
            )

        results: list[PlannerCandidateVerification] = []
        for idx, action in enumerate(candidates):
            result = self._checker.verify_by_state(current_state_id, action, goal_state)
            results.append(
                PlannerCandidateVerification(
                    index=idx,
                    action=action,
                    result=result,
                    selectable=result.result is VerifyResult.ALLOW,
                )
            )

        # The mask is the planner-time product; the aggregate tiers by the best
        # candidate: any selectable entry makes it ALLOW, any undecided candidate
        # keeps it UNCERTAIN, and only a set of pure proven refutations is DENY.
        if any(r.result.result is VerifyResult.ALLOW for r in results):
            overall = VerifyResult.ALLOW
        elif any(r.result.result is VerifyResult.UNCERTAIN for r in results):
            overall = VerifyResult.UNCERTAIN
        else:
            overall = VerifyResult.DENY

        return PlannerVerification(
            overall_result=overall,
            candidate_results=results,
            action_mask=[r.selectable for r in results],
        )

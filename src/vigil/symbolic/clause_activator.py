"""Runtime activation of independently admitted transition-guard clauses.

Activation is deliberately separate from DSL evaluation.  It consumes only
offline-admitted clause metadata plus the mobile agent's frozen value snapshot,
then selects the expressions that the existing :class:`DSLEvaluator` should
evaluate for this action.

A slot is *bound* iff its key is present in the snapshot.  There is no status
lifecycle: a missing key is simply unbound, and requiredness (admitted guard
semantics) decides whether that is skipped or inconclusive.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field

from vigil.models.guard import (
    GuardClause,
    GuardContract,
    IntentSlot,
    SlotRequiredness,
)
from vigil.models.intent_binding import (
    IntentBindingSnapshot,
    IntentBindingValue,
)
from vigil.symbolic.intent_patch import bound_value_error
from vigil.symbolic.numeric_normalizer import normalize_numeric_scalar


class ClauseDisposition(StrEnum):
    """Runtime disposition of one admitted guard clause."""

    ACTIVE = "active"
    SKIPPED = "skipped"
    INCONCLUSIVE = "inconclusive"


class ClauseActivationTrace(BaseModel):
    """Evidence record for one clause activation decision."""

    clause_index: int
    compiled_expr: str
    slot: str | None = None
    requiredness: SlotRequiredness | None = None
    slot_bound: bool | None = None
    disposition: ClauseDisposition
    reason: str


class ClauseActivationResult(BaseModel):
    """All clause dispositions plus the values available to the DSL evaluator."""

    clauses: list[ClauseActivationTrace] = Field(default_factory=list)
    bound_variables: dict[str, IntentBindingValue] = Field(default_factory=dict)
    # Populated when two ACTIVE numeric slots compared against the SAME source
    # scalar are bound an exact power of ten (>= 100x) apart — the signature of a
    # covert scale shift (cents vs dollars). Evaluating such a group would make
    # one clause a scale artifact (a vacuous cap or a spurious inequality), so
    # the decision engine withholds instead of evaluating it.
    scale_incoherence: list[str] = Field(default_factory=list)

    @property
    def active_expressions(self) -> list[str]:
        return [
            clause.compiled_expr
            for clause in self.clauses
            if clause.disposition is ClauseDisposition.ACTIVE
        ]

    @property
    def expression(self) -> str | None:
        expressions = self.active_expressions
        return " && ".join(expressions) if expressions else None

    @property
    def has_inconclusive(self) -> bool:
        return any(clause.disposition is ClauseDisposition.INCONCLUSIVE for clause in self.clauses)


class ClauseActivator:
    """Apply the required/optional value-presence truth table to admitted clauses."""

    def activate(
        self,
        contract: GuardContract,
        binding_snapshot: IntentBindingSnapshot,
    ) -> ClauseActivationResult:
        slots = {slot.name: slot for slot in contract.slots}
        traces: list[ClauseActivationTrace] = []
        bound_variables: dict[str, IntentBindingValue] = {}

        for index, clause in enumerate(contract.clauses):
            trace = self._activate_clause(index, clause, binding_snapshot, slots)
            traces.append(trace)
            if trace.disposition is ClauseDisposition.ACTIVE and trace.slot is not None:
                bound_variables[trace.slot] = binding_snapshot.bindings[trace.slot]

        return ClauseActivationResult(
            clauses=traces,
            bound_variables=bound_variables,
            scale_incoherence=self._scale_incoherence(contract, traces, bound_variables),
        )

    @staticmethod
    def _scale_incoherence(
        contract: GuardContract,
        traces: list[ClauseActivationTrace],
        bound_variables: dict[str, IntentBindingValue],
    ) -> list[str]:
        """Detect numeric slots bound in different scales against one source scalar.

        Groups ACTIVE slot-bound clauses by the source scalar their predicate reads
        (``read``/``value`` on one element property). Two DIFFERENT slots whose
        bound values sit an exact power of ten (>= 100x) apart cannot both be in
        the on-screen scale: one comparison may reflect inconsistent units.
        The group is reported so the decision engine can withhold and the
        evidence-only feedback can name every slot that needs consistent binding.

        A 100x ratio between two genuinely independent amounts is possible ($1
        item under a $100 budget) and is conservatively flagged too: the cost is
        a refusal that the agent resolves by rebinding, never a wrong ALLOW.
        """
        active_by_index = {
            trace.clause_index: trace
            for trace in traces
            if trace.disposition is ClauseDisposition.ACTIVE and trace.slot is not None
        }
        groups: dict[tuple[str, str, str], dict[str, Decimal]] = {}
        for index, clause in enumerate(contract.clauses):
            trace = active_by_index.get(index)
            if trace is None or trace.slot is None:
                continue
            predicate = clause.predicate
            if predicate.predicate_type not in ("read", "value"):
                continue
            value = bound_variables.get(trace.slot)
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            key = (
                predicate.predicate_type,
                predicate.element or "",
                predicate.property or "",
            )
            groups.setdefault(key, {})[trace.slot] = normalize_numeric_scalar(
                value, source_value=False
            )
        findings: list[str] = []
        for (_, element, _), slot_values in sorted(groups.items()):
            if len(slot_values) < 2:
                continue
            items = sorted(slot_values.items())
            for i, (slot_a, val_a) in enumerate(items):
                for slot_b, val_b in items[i + 1 :]:
                    if not val_a or not val_b:
                        continue
                    big, small = max(abs(val_a), abs(val_b)), min(abs(val_a), abs(val_b))
                    ratio = big / small
                    if ratio >= 100 and ratio == ratio.to_integral_value():
                        digits = str(ratio.to_integral_value())
                        if digits[0] == "1" and set(digits[1:]) <= {"0"}:
                            findings.append(
                                f"slots '{slot_a}'={val_a} and '{slot_b}'={val_b} "
                                f"compared against '{element}' differ by exactly "
                                f"{digits}x; rebind both in the on-screen scale"
                            )
        return findings

    @staticmethod
    def _activate_clause(
        index: int,
        clause: GuardClause,
        snapshot: IntentBindingSnapshot,
        slots: dict[str, IntentSlot],
    ) -> ClauseActivationTrace:
        base = {
            "clause_index": index,
            "compiled_expr": clause.compiled_expr,
            "slot": clause.slot,
            "requiredness": None,
        }

        if clause.slot is None:
            return ClauseActivationTrace(
                **base,
                disposition=ClauseDisposition.ACTIVE,
                reason="unconditional_clause",
            )

        slot = slots.get(clause.slot)
        if slot is None:
            return ClauseActivationTrace(
                **base,
                disposition=ClauseDisposition.INCONCLUSIVE,
                reason="clause_slot_not_declared",
            )
        requiredness = slot.requiredness
        base["requiredness"] = requiredness
        if requiredness is SlotRequiredness.UNKNOWN:
            return ClauseActivationTrace(
                **base,
                disposition=ClauseDisposition.INCONCLUSIVE,
                reason="slot_requiredness_not_admitted",
            )

        # A slot is bound iff its key is present; a present value may be falsy
        # (0, "", False), so test membership, never truthiness.
        bound = clause.slot in snapshot.bindings
        if bound:
            value = snapshot.bindings[clause.slot]
            value_error = bound_value_error(slot, value)
            if not value_error and clause.predicate.predicate_type == "count":
                normalized = normalize_numeric_scalar(value, source_value=False)
                if normalized < 0:
                    value_error = "bound_count_value_must_be_non_negative"
                elif normalized != normalized.to_integral_value():
                    value_error = "bound_count_value_must_be_integral"
            if value_error:
                return ClauseActivationTrace(
                    **base,
                    slot_bound=True,
                    disposition=ClauseDisposition.INCONCLUSIVE,
                    reason=value_error,
                )
            return ClauseActivationTrace(
                **base,
                slot_bound=True,
                disposition=ClauseDisposition.ACTIVE,
                reason="slot_bound",
            )
        if requiredness is SlotRequiredness.OPTIONAL:
            return ClauseActivationTrace(
                **base,
                slot_bound=False,
                disposition=ClauseDisposition.SKIPPED,
                reason="optional_slot_missing",
            )
        return ClauseActivationTrace(
            **base,
            slot_bound=False,
            disposition=ClauseDisposition.INCONCLUSIVE,
            reason="required_slot_missing",
        )

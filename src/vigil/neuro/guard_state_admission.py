"""Source-state guard admission (Stage 4, step 5).

The neural generator proposes one :class:`~vigil.models.guard.GuardContract` per
transition. This module raises the *symbolic admission unit* to a whole source state: it
takes the staged per-transition contracts for one source state, runs the existing
per-contract admission on each of them, and then enforces a single deterministic
cross-contract invariant before any contract may be attached.

The invariant (schema coherence, **not** semantic completeness): within one source state
``s``, for every canonical intent-slot name ``n``,

    F(s, n) = { intent_slot_fingerprint(slot)
                | an admitted executable outgoing clause from s references slot n }

the batch is coherent iff ``|F(s, n)| <= 1`` for every ``n``. This is the same fingerprint
and admitted-executable-clause predicate the runtime intent-patch derivation
(:func:`vigil.symbolic.intent_patch.derive_intent_patch_demand`) relies on, so an attached
batch can never violate it.

Admission is all-or-nothing per source state: a generation failure, a rejected contract, or
a coherence conflict rejects the whole batch, and the caller leaves every transition's
prior contract untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vigil.models.guard import GuardAdmissionStatus, GuardContract, IntentSlot, SlotType
from vigil.neuro.guard_admission import (
    GuardAdmissionResult,
    admit_guard_contract,
    apply_guard_admission,
)
from vigil.symbolic.intent_patch import intent_slot_fingerprint

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from vigil.neuro.guard_evidence import GuardEvidence


@dataclass(frozen=True)
class StateGuardCandidate:
    """One staged, pre-admission contract for a known outgoing transition.

    ``generation_failed`` records an explicit failed generation outcome (structured output
    unavailable, refusal, or validation failure). Its contract is never admitted.
    """

    transition_index: int
    evidence: GuardEvidence
    contract: GuardContract
    generation_failed: bool = False
    failure_reason: str = ""


@dataclass(frozen=True)
class AdmittedTransitionGuard:
    """A promotable admitted contract, built only when the whole batch admits."""

    transition_index: int
    contract: GuardContract
    guard: str | None
    result: GuardAdmissionResult


@dataclass(frozen=True)
class SlotSchemaSummary:
    """Exact, actionable schema witness for one (transition, slot-name) occurrence."""

    transition_index: int
    slot_name: str
    slot_type: str
    enum_domain: tuple[str, ...]
    fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return {
            "transition_index": self.transition_index,
            "slot_name": self.slot_name,
            "slot_type": self.slot_type,
            "enum_domain": list(self.enum_domain),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class SlotFingerprintConflict:
    """A canonical slot name referenced with divergent fingerprints within one source state.

    ``occurrences`` carries at most one summary per (transition, canonical slot name), so
    retry feedback exposes exactly which transitions disagree and how.
    """

    slot_name: str
    occurrences: tuple[SlotSchemaSummary, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "slot_name": self.slot_name,
            "fingerprints": sorted({occ.fingerprint for occ in self.occurrences}),
            "occurrences": [occ.as_dict() for occ in self.occurrences],
        }


@dataclass(frozen=True)
class StateBatchDiagnostics:
    """Deterministic, machine-readable diagnostics for one source-state attempt."""

    source_state_id: str
    generation_failures: list[dict[str, object]] = field(default_factory=list)
    admission_failures: list[dict[str, object]] = field(default_factory=list)
    coherence_conflicts: list[SlotFingerprintConflict] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "source_state_id": self.source_state_id,
            "generation_failures": list(self.generation_failures),
            "admission_failures": list(self.admission_failures),
            "coherence_conflicts": [conflict.as_dict() for conflict in self.coherence_conflicts],
        }


@dataclass(frozen=True)
class StateBatchOutcome:
    """Outcome of admitting one source state's staged contract batch.

    ``results`` is the promotable projection, empty unless ``admitted``.
    """

    admitted: bool
    source_state_id: str
    results: list[AdmittedTransitionGuard]
    diagnostics: StateBatchDiagnostics


def _slot_schema_summary(transition_index: int, slot: IntentSlot) -> SlotSchemaSummary:
    enum_domain = tuple(sorted(slot.value_domain)) if slot.slot_type is SlotType.ENUM else ()
    return SlotSchemaSummary(
        transition_index=transition_index,
        slot_name=slot.name.strip(),
        slot_type=slot.slot_type.value,
        enum_domain=enum_domain,
        fingerprint=intent_slot_fingerprint(slot),
    )


def check_slot_fingerprint_coherence(
    admitted: Sequence[tuple[int, GuardContract]],
) -> list[SlotFingerprintConflict]:
    """Same-name slot-fingerprint coherence over admitted contracts of one source state.

    Within each contract, referenced canonical slot names are deduplicated (one occurrence
    per transition/name) using the admitted-executable-clause predicate
    (``admission_status is ADMITTED and slot is not None and compiled_expr``); a name whose
    fingerprint set has size > 1 across transitions is a conflict.
    """
    occurrences: dict[str, list[SlotSchemaSummary]] = {}
    fingerprints: dict[str, set[str]] = {}
    for transition_index, contract in admitted:
        declared = {slot.name: slot for slot in contract.slots}
        referenced: set[str] = set()
        for clause in contract.clauses:
            if (
                clause.admission_status is GuardAdmissionStatus.ADMITTED
                and clause.slot is not None
                and clause.compiled_expr
            ):
                referenced.add(clause.slot)
        for name in referenced:
            summary = _slot_schema_summary(transition_index, declared[name])
            occurrences.setdefault(name, []).append(summary)
            fingerprints.setdefault(name, set()).add(summary.fingerprint)

    conflicts = [
        SlotFingerprintConflict(
            slot_name=name,
            occurrences=tuple(
                sorted(occurrences[name], key=lambda occ: (occ.transition_index, occ.fingerprint))
            ),
        )
        for name, fps in fingerprints.items()
        if len(fps) > 1
    ]
    return sorted(conflicts, key=lambda conflict: conflict.slot_name)


def admit_source_state_batch(
    candidates: Sequence[StateGuardCandidate],
    *,
    source_state_id: str,
) -> StateBatchOutcome:
    """Admit one source state's staged contract batch atomically (all-or-nothing).

    Steps: (1) run per-contract admission on independent deep copies for every candidate;
    (2) reject the whole batch if any candidate failed generation or per-contract
    admission; (3) else run the same-name slot-fingerprint coherence invariant; (4) admit
    only when every step passes. Input contracts are never mutated; attachment is the
    caller's responsibility.
    """
    generation_failures: list[dict[str, object]] = []
    admission_failures: list[dict[str, object]] = []
    admitted_pairs: list[tuple[int, GuardContract]] = []
    promotable: list[AdmittedTransitionGuard] = []

    for candidate in candidates:
        if candidate.generation_failed:
            reason = candidate.failure_reason or "structured guard generation failed"
            generation_failures.append(
                {"transition_index": candidate.transition_index, "reason": reason}
            )
            continue

        working = candidate.contract.model_copy(deep=True)
        result = admit_guard_contract(working, candidate.evidence)
        if result.admitted:
            admitted_contract = apply_guard_admission(working, result)
            admitted_pairs.append((candidate.transition_index, admitted_contract))
            promotable.append(
                AdmittedTransitionGuard(
                    transition_index=candidate.transition_index,
                    contract=admitted_contract,
                    guard=result.guard,
                    result=result,
                )
            )
        else:
            admission_failures.append(
                {
                    "transition_index": candidate.transition_index,
                    "reason": result.reason,
                    "rejected_predicates": list(result.rejected_predicates),
                }
            )

    coherence_conflicts: list[SlotFingerprintConflict] = []
    admitted = not (generation_failures or admission_failures)
    if admitted:
        coherence_conflicts = check_slot_fingerprint_coherence(admitted_pairs)
        if coherence_conflicts:
            admitted = False

    diagnostics = StateBatchDiagnostics(
        source_state_id=source_state_id,
        generation_failures=generation_failures,
        admission_failures=admission_failures,
        coherence_conflicts=coherence_conflicts,
    )
    return StateBatchOutcome(
        admitted=admitted,
        source_state_id=source_state_id,
        results=promotable if admitted else [],
        diagnostics=diagnostics,
    )

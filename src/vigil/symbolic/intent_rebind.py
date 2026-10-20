"""Evidence-triggered intent rebinding admission.

When a real ``DENY / guard_failed`` proves an active BOUND intent slot made a
guard predicate false, Vigil issues a typed :class:`RecoveryAuthorization` naming
only those slots. The mobile agent may then propose a revised value for an
eligible slot; :func:`admit_intent_rebind` validates the revision, schema,
fingerprint, type, and enum domain, and — unlike an ordinary add-only patch —
overwrites the eligible key, freezing a new immutable snapshot revision when the
value changes. A same-value proposal is a no-op that keeps the revision.

Rebinding never returns a verdict. Vigil never chooses or derives the value. The
adapter is stateless, so eligibility is re-derived structurally from the live FSM
and snapshot here; the fact that the predicate was false originates from the exact
verifier witness carried on the authorization (a trusted, non-adversarial harness
transports it — no tamper-proof claim is made). The per-chain attempt budget is
owned by the single recovery ledger, not by this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from vigil.models.fsm import AppFSM, TransitionLookupStatus, canonical_action_key
from vigil.models.guard import GuardAdmissionStatus
from vigil.models.intent_binding import (
    AdmittedPlannerProposal,
    IntentBindingSnapshot,
    IntentBindingValue,
    IntentPatch,
    RevisionedIntentBindingSnapshot,
)
from vigil.symbolic.clause_activator import ClauseActivationResult, ClauseDisposition
from vigil.symbolic.dsl_evaluator import GuardStatus, PredicateWitness
from vigil.symbolic.intent_patch import (
    bound_value_error,
    canonical_agent_action_proposal,
    intent_slot_fingerprint,
)


class EligibleSlot(BaseModel):
    """One rebind-eligible slot: its canonical name and admitted fingerprint.

    ``observed_value`` is the screen literal the failing predicate actually read
    for this slot's clause. It is a *presentation hint* only: it never enters the
    recovery chain identity, admission, or any structural eligibility check, and
    Vigil never proposes it as the revised value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: StrictStr
    fingerprint: StrictStr
    observed_value: StrictStr | None = None


class RecoveryAuthorization(BaseModel):
    """A typed permission to revise specific BOUND slots after a guard failure.

    Issued only for existing snapshot keys whose admitted, active clause's
    predicate was ``false`` in the real evaluator witness. It permits — but never
    requires — a rebind, and is bound to the exact failing action and source
    state so a later attempt cannot drift to a different transition.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: StrictStr
    efsm_revision: StrictInt = Field(ge=0)
    recovery_chain_id: StrictStr
    source_state_id: StrictStr
    failed_action: dict[str, Any]
    eligible_slots: tuple[EligibleSlot, ...]

    @property
    def eligible_fingerprints(self) -> dict[str, str]:
        return {slot.name: slot.fingerprint for slot in self.eligible_slots}


class IntentRebindAdmissionResult(BaseModel):
    """Deterministic outcome of one recovery rebind admission (never a verdict)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    admitted: bool
    admitted_proposal: AdmittedPlannerProposal | None = None
    reasons: tuple[str, ...] = ()
    no_op: bool = False


def recovery_chain_id(
    *,
    efsm_revision: int,
    source_state_id: str,
    failed_action: Mapping[str, Any],
    eligible: Sequence[tuple[str, str]],
) -> str:
    """Deterministic, revision-independent identity for one recovery chain.

    Keyed on the canonical action identity (value payload stripped) plus the
    sorted eligible ``(name, fingerprint)`` pairs, so the chain survives
    intervening ordinary planner rounds that only bump the snapshot revision.
    """
    payload = {
        "efsm_revision": efsm_revision,
        "source_state_id": source_state_id,
        "action": str(canonical_action_key(dict(failed_action))),
        "eligible": sorted((name, fingerprint) for name, fingerprint in eligible),
    }
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "rchain:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _observed_literal(value: Any) -> str | None:
    """Render one witness ``actual`` scalar as a display string (never a value decision)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def derive_recovery_eligibility(
    *,
    activation: ClauseActivationResult,
    witness: Sequence[PredicateWitness] | None,
    snapshot: RevisionedIntentBindingSnapshot,
) -> list[EligibleSlot]:
    """Eligible BOUND slots from a real guard-failure witness.

    A slot is eligible iff its clause is admitted and ACTIVE, its slot is
    non-null, the predicate result is ``false`` in the real pass, and the value is
    present with a recorded fingerprint. Positional attribution requires one
    predicate per active clause; on any cardinality mismatch nothing is eligible.

    Each eligible slot also carries the failing predicate's ``actual`` screen
    literal as ``observed_value``, a presentation hint that never affects
    eligibility, chain identity, or admission.
    """
    active = [
        clause for clause in activation.clauses if clause.disposition is ClauseDisposition.ACTIVE
    ]
    records = list(witness or ())
    if len(active) != len(records):
        return []
    eligible: dict[str, tuple[str, str | None]] = {}
    for clause, record in zip(active, records, strict=True):
        if clause.slot is None or record.result is not GuardStatus.FALSE:
            continue
        if clause.slot not in snapshot.values.bindings:
            continue
        fingerprint = snapshot.slot_fingerprints.get(clause.slot)
        if fingerprint is None:
            continue
        eligible[clause.slot] = (fingerprint, _observed_literal(record.actual))
    return [
        EligibleSlot(name=name, fingerprint=fingerprint, observed_value=observed)
        for name, (fingerprint, observed) in sorted(eligible.items())
    ]


def build_recovery_authorization(
    *,
    event_id: str,
    efsm_revision: int,
    source_state_id: str,
    failed_action: Mapping[str, Any],
    activation: ClauseActivationResult,
    witness: Sequence[PredicateWitness] | None,
    snapshot: RevisionedIntentBindingSnapshot,
) -> RecoveryAuthorization | None:
    """Issue an authorization for a guard failure, or ``None`` when none applies."""
    eligible = derive_recovery_eligibility(
        activation=activation, witness=witness, snapshot=snapshot
    )
    if not eligible:
        return None
    chain_id = recovery_chain_id(
        efsm_revision=efsm_revision,
        source_state_id=source_state_id,
        failed_action=failed_action,
        eligible=[(slot.name, slot.fingerprint) for slot in eligible],
    )
    return RecoveryAuthorization(
        event_id=event_id,
        efsm_revision=efsm_revision,
        recovery_chain_id=chain_id,
        source_state_id=source_state_id,
        failed_action=dict(failed_action),
        eligible_slots=tuple(eligible),
    )


def _structurally_eligible(
    fsm: AppFSM,
    authorization: RecoveryAuthorization,
    base_snapshot: RevisionedIntentBindingSnapshot,
) -> tuple[dict[str, Any], list[str]]:
    """Re-derive the eligible slots from the live FSM + snapshot.

    Returns ``(declared_slots_by_name, reasons)``. ``reasons`` is non-empty when
    the authorization cannot be matched against the current admitted contract.
    """
    reasons: list[str] = []
    lookup = fsm.resolve_transition(authorization.source_state_id, authorization.failed_action)
    if lookup.status is not TransitionLookupStatus.MATCH or lookup.transition is None:
        return {}, ["failed_transition_not_found"]
    contract = lookup.transition.guard_contract
    if contract is None or contract.admission_status is not GuardAdmissionStatus.ADMITTED:
        return {}, ["no_admitted_contract"]
    declared = {slot.name: slot for slot in contract.slots}
    referenced = {
        clause.slot
        for clause in contract.clauses
        if clause.admission_status is GuardAdmissionStatus.ADMITTED
        and clause.slot is not None
        and clause.compiled_expr
    }
    live_eligible: dict[str, str] = {}
    for name in referenced:
        slot = declared.get(name)
        if slot is None:
            continue
        fingerprint = intent_slot_fingerprint(slot)
        if (
            name in base_snapshot.values.bindings
            and base_snapshot.slot_fingerprints.get(name) == fingerprint
        ):
            live_eligible[name] = fingerprint

    for slot in authorization.eligible_slots:
        if live_eligible.get(slot.name) != slot.fingerprint:
            reasons.append(f"authorization_slot_not_eligible:{slot.name}")

    return declared, reasons


def admit_intent_rebind(
    *,
    fsm: AppFSM,
    base_snapshot: RevisionedIntentBindingSnapshot,
    patch: IntentPatch,
    authorization: RecoveryAuthorization,
    agent_action_proposal: Mapping[str, Any],
) -> IntentRebindAdmissionResult:
    """Admit one evidence-triggered rebind and freeze the resulting revision.

    Overwrites only the authorization's eligible keys. A real value change bumps
    the revision by one; a same-value proposal is a no-op that keeps the revision
    (the ledger still consumes an attempt). Never returns a verdict.
    """
    reasons: list[str] = []
    if authorization.efsm_revision != fsm.efsm_revision:
        reasons.append(f"efsm_revision_mismatch:{authorization.efsm_revision}!={fsm.efsm_revision}")
    if patch.base_revision != base_snapshot.revision:
        reasons.append(f"base_revision_mismatch:{patch.base_revision}!={base_snapshot.revision}")

    declared, correlation_reasons = _structurally_eligible(fsm, authorization, base_snapshot)
    reasons.extend(correlation_reasons)

    eligible = authorization.eligible_fingerprints
    proposed = set(patch.entries)
    for name in sorted(proposed - set(eligible)):
        reasons.append(f"unlisted_slot:{name}")
    for name in sorted(proposed & set(eligible)):
        slot = declared.get(name)
        if slot is None:
            continue
        error = bound_value_error(slot, patch.entries[name])
        if error:
            reasons.append(f"{error}:{name}")

    if reasons:
        return IntentRebindAdmissionResult(admitted=False, reasons=tuple(reasons))

    merged: dict[str, IntentBindingValue] = dict(base_snapshot.values.bindings)
    for name, value in patch.entries.items():
        merged[name] = value
    values = IntentBindingSnapshot.model_validate(merged)
    changed = values != base_snapshot.values
    admitted_snapshot = (
        RevisionedIntentBindingSnapshot(
            revision=base_snapshot.revision + 1,
            values=values,
            slot_fingerprints=dict(base_snapshot.slot_fingerprints),
        )
        if changed
        else base_snapshot
    )
    canonical_action = canonical_agent_action_proposal(agent_action_proposal)
    proposal_id = (
        f"rebind:{authorization.recovery_chain_id}:m{authorization.efsm_revision}:"
        f"r{admitted_snapshot.revision}:{canonical_action}"
    )
    return IntentRebindAdmissionResult(
        admitted=True,
        no_op=not changed,
        admitted_proposal=AdmittedPlannerProposal(
            proposal_id=proposal_id,
            request_id=authorization.event_id,
            efsm_revision=authorization.efsm_revision,
            agent_action_proposal=canonical_action,
            binding_snapshot=admitted_snapshot,
        ),
    )

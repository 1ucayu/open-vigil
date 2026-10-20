"""Agent-agnostic admission for mobile-agent intent patches.

Mobile-agent adapters own natural-language interpretation and emit one status-aware
patch beside a normal planner action proposal. Vigil verifies the base revision,
current-state demand, update lifecycle, and offline-admitted slot schemas. Vigil does
not judge the natural-language interpretation. An admitted patch binds the opaque
agent action proposal to one immutable revisioned snapshot; it never changes the EFSM
or chooses an action.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vigil.models.fsm import Transition, canonical_transition_action
from vigil.models.guard import GuardAdmissionStatus, IntentSlot, SlotRequiredness, SlotType
from vigil.models.intent_binding import (
    AdmittedPlannerProposal,
    IntentBindingSnapshot,
    IntentBindingValue,
    IntentPatch,
    IntentPatchAdmissionResult,
    RevisionedIntentBindingSnapshot,
)
from vigil.symbolic.numeric_normalizer import (
    NumericNormalizationError,
    normalize_numeric_scalar,
)


def bound_value_error(slot: IntentSlot, value: object) -> str:
    """Return the schema violation for one bound value, or an empty string."""
    if slot.slot_type is SlotType.STRING:
        return "" if isinstance(value, str) else "bound_value_type_mismatch"
    if slot.slot_type is SlotType.BOOLEAN:
        return "" if isinstance(value, bool) else "bound_value_type_mismatch"
    if slot.slot_type is SlotType.ENUM:
        if not isinstance(value, str):
            return "bound_value_type_mismatch"
        if slot.value_domain and value not in slot.value_domain:
            return "bound_enum_value_outside_domain"
        return ""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "bound_value_type_mismatch"
    try:
        normalize_numeric_scalar(value, source_value=False)
    except NumericNormalizationError:
        return "bound_numeric_value_invalid"
    return ""


_GOAL_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _goal_numbers(goal_text: str) -> set[str]:
    """Decimal-normalized string forms of every number readable in the goal text."""
    from decimal import Decimal

    numbers: set[str] = set()
    for token in _GOAL_NUMBER_RE.findall(goal_text):
        value = Decimal(token)
        # A money amount is derivable in both renderings the apps use: the
        # written form ($85.00 -> 85) and integer minor units (8500). This adds
        # no unit ontology to the verifier: both candidates come verbatim from
        # one arithmetic relation on a number the user actually wrote.
        numbers.add(str(value.normalize()))
        numbers.add(str((value * 100).normalize()))
        if value == value.to_integral_value() and value % 100 == 0:
            numbers.add(str((value / 100).normalize()))
    return numbers


def ungrounded_value_error(
    slot: IntentSlot,
    value: object,
    goal_text: str,
    screen_texts: Sequence[str] = (),
) -> str:
    """Return why a bound value is derivable from neither goal nor screen, or "".

    The agent authors ``$intent.*`` bindings itself, so schema-only admission
    lets it satisfy any guard by inventing values the user never asked for
    (documented escape mechanism: ungrounded agent-authored patches). This gate
    is deterministic and judges derivability against the two legitimate sources
    a binding can quote — the user's goal and the live screen the patch
    accompanies (screen attestations such as a masked account number or an
    on-screen order total are correct bindings that never appear in goal text):

    - enum slots are exempt (the admitted value domain is offline evidence);
    - booleans are exempt (polarity is legitimately implicit in phrasing);
    - a number must equal a number readable in the goal or on the screen, in or
      out of minor units (85 / 8500 for "$85.00");
    - a string must appear case-insensitively in the goal or in a screen text.

    It deliberately does NOT judge which source value belongs in which slot —
    that is semantic interpretation, which stays with the agent.
    """
    if not goal_text and not screen_texts:
        return ""
    if slot.slot_type in (SlotType.ENUM, SlotType.BOOLEAN):
        return ""
    if slot.slot_type is SlotType.STRING:
        if not isinstance(value, str):
            return ""
        needle = value.strip().casefold()
        if needle in goal_text.casefold():
            return ""
        if any(needle and needle in text.casefold() for text in screen_texts):
            return ""
        return "ungrounded_string_value"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return ""
    try:
        normalized = str(normalize_numeric_scalar(value, source_value=False).normalize())
    except NumericNormalizationError:
        return ""
    universe = _goal_numbers(goal_text)
    for text in screen_texts:
        universe |= _goal_numbers(text)
    if normalized not in universe:
        return "ungrounded_numeric_value"
    return ""


def intent_slot_fingerprint(slot: IntentSlot) -> str:
    """Hash the stable value semantics/schema of one admitted intent slot.

    Natural-language description, requiredness, transition/widget context,
    evidence, and compiled predicates are intentionally excluded: they govern
    prompting, activation, or usage, not the machine-checked identity of the
    task value, and therefore must not prevent cross-transition reuse.
    """
    payload = {
        "name": slot.name.strip(),
        "type": slot.slot_type.value,
        "enum_domain": (sorted(slot.value_domain) if slot.slot_type is SlotType.ENUM else []),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Attestation shapes: the guard compares one concrete source-screen surface with the
# frozen slot under an *echo* operator, so the live on-screen literal is the canonical
# satisfying candidate. Constraint operators (``!=``, ``<``, ``>=``, ``not_contains``,
# ...) deliberately do not match: transcribing the screen would refute them.
_ELEMENT_PATTERN = r"[A-Za-z_][A-Za-z0-9_.:/]*"
_PROPERTY_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
_SLOT_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
_ECHO_OPERATOR_PATTERN = r"==|contains"

_READ_ATTESTATION_RE = re.compile(
    rf"^read\(\s*(?P<element>{_ELEMENT_PATTERN})\s*,\s*(?P<property>{_PROPERTY_PATTERN})\s*\)"
    rf"\s*(?:{_ECHO_OPERATOR_PATTERN})\s*\$intent\.(?P<slot>{_SLOT_PATTERN})$"
)
_VALUE_ATTESTATION_RE = re.compile(
    rf"^value\(\s*(?P<element>{_ELEMENT_PATTERN})\s*\)"
    rf"\s*(?:{_ECHO_OPERATOR_PATTERN})\s*\$intent\.(?P<slot>{_SLOT_PATTERN})$"
)


def _clause_screen_surface(clause: Mapping[str, Any], slot_name: str) -> tuple[str, str] | None:
    """Return the ``(element, property)`` this clause attests, or ``None``.

    Purely syntactic over the admitted compiled expression — the exact text the
    runtime evaluator re-checks against the live screen.
    """
    expression = str(clause.get("expression") or "")
    predicate_type = clause.get("predicate_type")
    if predicate_type == "read":
        match = _READ_ATTESTATION_RE.match(expression)
        if match is not None and match.group("slot") == slot_name:
            return match.group("element"), match.group("property")
        return None
    if predicate_type == "value":
        match = _VALUE_ATTESTATION_RE.match(expression)
        if match is not None and match.group("slot") == slot_name:
            return match.group("element"), "value"
        return None
    return None


def _slot_evidence_surface(
    transition_usage: Sequence[Mapping[str, Any]], slot_name: str
) -> dict[str, Any]:
    """Classify where one demanded slot's value is evidenced.

    ``screen_attestation`` requires *every* clause referencing the slot to compare
    the same concrete source-screen surface with the slot under an echo operator;
    anything else (action predicates, mixed surfaces, constraint operators) is an
    ``instruction`` slot whose value may only come from the user instruction.

    This is presentation metadata only: it never enters the slot fingerprint and
    never affects admission.
    """
    surfaces: set[tuple[str, str]] = set()
    for usage in transition_usage:
        for clause in usage["clauses"]:
            surface = _clause_screen_surface(clause, slot_name)
            if surface is None:
                return {"role": "instruction"}
            surfaces.add(surface)
    if len(surfaces) != 1:
        return {"role": "instruction"}
    element, prop = surfaces.pop()
    return {"element": element, "property": prop, "role": "screen_attestation"}


def _slot_requiredness(transition_usage: Sequence[Mapping[str, Any]]) -> str:
    """Machine-derived requiredness of one demanded slot across its usages.

    A slot is ``required`` iff at least one clause usage declares it required;
    ``unknown``/``evidence_insufficient`` requiredness degrades to ``optional``.
    """
    required = any(
        usage["requiredness"] == SlotRequiredness.REQUIRED.value for usage in transition_usage
    )
    return SlotRequiredness.REQUIRED.value if required else SlotRequiredness.OPTIONAL.value


@dataclass(frozen=True)
class IntentPatchDemand:
    """Canonical current-state demand consumed by every mobile-agent adapter.

    ``requested_slots`` in the request lists the slots that are missing or whose
    admitted schema/fingerprint changed. It is prompt guidance, not a hard
    obligation: a patch may omit a requested key, leaving that slot missing
    (runtime returns UNCERTAIN when an active required clause needs it).
    """

    request: dict[str, Any]
    efsm_revision: int
    slot_schemas: dict[str, IntentSlot]
    slot_fingerprints: dict[str, str]


def derive_intent_patch_demand(
    *,
    state_id: str,
    outgoing: Sequence[Transition],
    selectable: Sequence[bool],
    snapshot: RevisionedIntentBindingSnapshot,
    efsm_revision: int,
) -> IntentPatchDemand:
    """Build one canonical patch request from selectable admitted clauses.

    A slot whose key is present with the same fingerprint stays frozen (add-only:
    it cannot be overwritten by an ordinary patch). Every other referenced slot —
    missing, or present with a changed fingerprint — is a requested update.
    """
    records: dict[str, dict[str, Any]] = {}
    schemas: dict[str, IntentSlot] = {}
    fingerprints: dict[str, str] = {}

    for transition, flag in zip(outgoing, selectable, strict=True):
        contract = transition.guard_contract
        if (
            not flag
            or contract is None
            or contract.admission_status is not GuardAdmissionStatus.ADMITTED
        ):
            continue
        declared = {slot.name: slot for slot in contract.slots}
        referenced: dict[str, list[tuple[int, Any]]] = {}
        for clause_index, clause in enumerate(contract.clauses):
            if (
                clause.admission_status is GuardAdmissionStatus.ADMITTED
                and clause.slot is not None
                and clause.compiled_expr
            ):
                referenced.setdefault(clause.slot, []).append((clause_index, clause))

        for name, clause_refs in referenced.items():
            slot = declared[name]
            fingerprint = intent_slot_fingerprint(slot)
            previous_fingerprint = fingerprints.get(name)
            if previous_fingerprint is not None and previous_fingerprint != fingerprint:
                raise ValueError(f"conflicting admitted semantics/schema for intent slot {name!r}")
            schemas.setdefault(name, slot)
            fingerprints.setdefault(name, fingerprint)

            record = records.setdefault(
                name,
                {
                    "name": name,
                    "type": slot.slot_type.value,
                    "description": slot.description,
                    "value_domain": list(slot.value_domain),
                    "slot_fingerprint": fingerprint,
                    "current_bound": name in snapshot.values.bindings,
                    "current_slot_fingerprint": snapshot.slot_fingerprints.get(name),
                    "transition_usage": [],
                },
            )
            record["transition_usage"].append(
                {
                    "from": transition.source,
                    "to": transition.target,
                    "action": canonical_transition_action(transition.action),
                    "requiredness": slot.requiredness.value,
                    "clauses": [
                        {
                            "index": clause_index,
                            "expression": clause.compiled_expr,
                            "predicate_type": clause.predicate.predicate_type,
                            "operator": clause.predicate.operator,
                        }
                        for clause_index, clause in clause_refs
                    ],
                }
            )

    requested_slots: list[dict[str, Any]] = []
    for name in sorted(records):
        record = records[name]
        present = name in snapshot.values.bindings
        current_fingerprint = snapshot.slot_fingerprints.get(name)
        fingerprint = fingerprints[name]

        # Present + same fingerprint ⇒ frozen (add-only, never re-requested).
        if present and current_fingerprint == fingerprint:
            continue
        requiredness = _slot_requiredness(record["transition_usage"])
        record["requiredness"] = requiredness
        record["evidence_surface"] = _slot_evidence_surface(record["transition_usage"], name)
        # Envelope compatibility: ``update_required`` now mirrors the derived
        # requiredness instead of unconditionally presenting every demand as required.
        record["update_required"] = requiredness == SlotRequiredness.REQUIRED.value
        record["update_reason"] = "slot_fingerprint_changed" if present else "missing"
        requested_slots.append(record)
    requested_names = [record["name"] for record in requested_slots]
    request_id = f"ipr:{state_id}:m{efsm_revision}:r{snapshot.revision}:" + (
        "|".join(requested_names) if requested_names else "-"
    )
    return IntentPatchDemand(
        request={
            "request_id": request_id,
            "state_id": state_id,
            "efsm_revision": efsm_revision,
            "base_revision": snapshot.revision,
            "requested_slots": requested_slots,
        },
        efsm_revision=efsm_revision,
        slot_schemas=schemas,
        slot_fingerprints=fingerprints,
    )


def canonical_agent_action_proposal(action: Mapping[str, Any]) -> str:
    """Return stable opaque correlation text for one agent-specific action."""
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def admit_intent_patch(
    *,
    base_snapshot: RevisionedIntentBindingSnapshot,
    patch: IntentPatch,
    demand: IntentPatchDemand,
    agent_action_proposal: Mapping[str, Any],
    goal_text: str | None = None,
    screen_texts: Sequence[str] = (),
) -> IntentPatchAdmissionResult:
    """Admit one planner-time delta and freeze the resulting snapshot revision.

    Admission is add-only: a slot may be supplied for a demanded key, but a key
    already present with the same fingerprint cannot be overwritten. A patch may
    omit a requested key (that slot stays missing). Validation covers the base
    revision, canonical slot identity, value type, and enum domain — plus, when
    the caller supplies ``goal_text``, value derivability from the user's goal
    (:func:`ungrounded_value_error`). Vigil still does not interpret natural
    language: derivability is a deterministic containment/arithmetic check, and
    with ``goal_text=None`` behavior is unchanged. A patch that changes nothing
    keeps the same snapshot revision.
    """
    allowed = set(demand.slot_schemas)
    proposed = set(patch.entries)
    reasons: list[str] = []

    if patch.base_revision != base_snapshot.revision:
        reasons.append(f"base_revision_mismatch:{patch.base_revision}!={base_snapshot.revision}")
    for name in sorted(proposed - allowed):
        reasons.append(f"undeclared_patch_slot:{name}")

    for name in sorted(proposed & set(base_snapshot.values.bindings)):
        same_fingerprint = base_snapshot.slot_fingerprints[name] == demand.slot_fingerprints.get(
            name
        )
        if same_fingerprint:
            reasons.append(f"slot_already_frozen:{name}")

    for name in sorted(proposed & allowed):
        slot = demand.slot_schemas[name]
        error = bound_value_error(slot, patch.entries[name])
        if error:
            reasons.append(f"{error}:{name}")
            continue
        grounding_error = ungrounded_value_error(
            slot, patch.entries[name], goal_text or "", screen_texts
        )
        if grounding_error:
            reasons.append(f"{grounding_error}:{name}")

    if reasons:
        return IntentPatchAdmissionResult(
            admitted=False,
            reasons=tuple(reasons),
        )

    merged: dict[str, IntentBindingValue] = dict(base_snapshot.values.bindings)
    for name, value in patch.entries.items():
        merged[name] = value
    values = IntentBindingSnapshot.model_validate(merged)
    slot_fingerprints = dict(base_snapshot.slot_fingerprints)
    for name in patch.entries:
        slot_fingerprints[name] = demand.slot_fingerprints[name]
    changed = values != base_snapshot.values or slot_fingerprints != dict(
        base_snapshot.slot_fingerprints
    )
    admitted_snapshot = (
        RevisionedIntentBindingSnapshot(
            revision=base_snapshot.revision + 1,
            values=values,
            slot_fingerprints=slot_fingerprints,
        )
        if changed
        else base_snapshot
    )
    canonical_action = canonical_agent_action_proposal(agent_action_proposal)
    request_id = str(demand.request["request_id"])
    proposal_id = (
        f"proposal:{request_id}:m{demand.efsm_revision}:"
        f"r{admitted_snapshot.revision}:{canonical_action}"
    )
    return IntentPatchAdmissionResult(
        admitted=True,
        admitted_proposal=AdmittedPlannerProposal(
            proposal_id=proposal_id,
            request_id=request_id,
            efsm_revision=demand.efsm_revision,
            agent_action_proposal=canonical_action,
            binding_snapshot=admitted_snapshot,
        ),
    )

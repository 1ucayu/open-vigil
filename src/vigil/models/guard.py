"""Typed guard-contract models for contract-first DSL guard generation.

These models are synthesis intermediate representation (IR) / metadata for the
guard-generation pipeline:

    existing FSM + trace evidence -> transition evidence view -> widget registry
    -> typed GuardContract -> DSL compilation -> admission validation
    -> attach guard metadata to the FSM

A ``GuardContract`` describes the intended pre-action guard for a transition in a
typed, compiler-friendly form. Runtime activation consumes only its independently
admitted clauses; ``Transition.guard`` is the compiled aggregate/audit view. This module
deliberately contains no registry, compiler, admission validator, or LLM logic — only the
data models that later stages fill in and consume.

LLM-facing candidates are strict; deterministic admission owns compiled expressions and
status metadata.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GuardKind(StrEnum):
    """Category of guard a contract represents."""

    NONE = "none"
    NAVIGATION = "navigation"
    ITEM_BINDING = "item_binding"
    INPUT_BINDING = "input_binding"
    TOGGLE_BINDING = "toggle_binding"
    FORM_CHECK = "form_check"
    CONFIRM_COMMIT = "confirm_commit"
    SAFETY_CHECK = "safety_check"
    UNKNOWN = "unknown"


class SlotType(StrEnum):
    """Value type of an intent slot."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ENUM = "enum"


class SlotRequiredness(StrEnum):
    """Counterfactual activation class of an intent-bound guard clause."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    UNKNOWN = "unknown"


class CounterfactualOutcome(StrEnum):
    """Effect of removing a slot while holding the transition evidence fixed."""

    TARGET_UNDERDETERMINED = "target_underdetermined"
    PAYLOAD_UNDERDETERMINED = "payload_underdetermined"
    EFFECT_UNDERDETERMINED = "effect_underdetermined"
    ACTION_REMAINS_VALID = "action_remains_valid"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"


RequirednessEvidenceSource = Literal[
    "known_action",
    "source_widget",
    "source_target_diff",
    "effect_metadata",
    "sibling_action",
    "trace",
]


class RequirednessEvidenceRef(BaseModel):
    """Reference to one fact in the guard evidence packet.

    ``source`` is a framework-level evidence namespace, not an application domain.
    ``identifier`` must resolve in the concrete :class:`GuardEvidence` packet during
    admission; an LLM-authored free-form explanation is intentionally not evidence.
    """

    model_config = ConfigDict(extra="forbid")

    source: RequirednessEvidenceSource
    identifier: str


class RequirednessEvidence(BaseModel):
    """Structured counterfactual evidence for one clause's requiredness."""

    model_config = ConfigDict(extra="forbid")

    outcome: CounterfactualOutcome
    refs: list[RequirednessEvidenceRef] = Field(default_factory=list)


class GuardAdmissionStatus(StrEnum):
    """Lifecycle status of a guard contract through admission validation."""

    PENDING = "pending"
    ADMITTED = "admitted"
    REJECTED = "rejected"


class IntentSlot(BaseModel):
    """A frozen ``$intent.*`` variable the guard may reference.

    Slots describe typed task values used by guard clauses. They are populated during
    contract synthesis and resolved against the contract slot schema during admission.
    """

    name: str
    slot_type: SlotType = SlotType.STRING
    description: str = ""
    requiredness: SlotRequiredness = SlotRequiredness.UNKNOWN
    requiredness_evidence: RequirednessEvidence = Field(
        default_factory=lambda: RequirednessEvidence(
            outcome=CounterfactualOutcome.EVIDENCE_INSUFFICIENT
        )
    )
    value_domain: list[str] = Field(default_factory=list)


class ValueRef(BaseModel):
    """A reference to a value used on one side of a predicate.

    ``kind`` selects the only two RHS sources admitted by the typed contract:

    - ``literal``: a constant in ``value``.
    - ``intent``: a frozen intent slot named by ``slot``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["literal", "intent"]
    value: Any | None = None
    slot: str | None = None


GuardPredicateType = Literal[
    "read",
    "value",
    "action",
    "count",
]


class PredicateSpec(BaseModel):
    """A single typed predicate within a guard contract.

    This is the contract-level analogue of one DSL predicate. ``predicate_type``
    mirrors the supported contract vocabulary; the compiler lowers executable
    ``PredicateSpec`` objects into concrete DSL syntax during the compilation pass.
    Text containment is expressed by a ``value`` or ``read`` predicate whose
    ``operator`` is ``contains``; it is not a separate predicate type.
    """

    predicate_type: GuardPredicateType
    element: str | None = None
    property: str | None = None
    operator: str | None = None
    expected: ValueRef | None = None


class GuardClause(BaseModel):
    """One independently compiled and admitted transition-guard predicate.

    ``slot=None`` denotes a literal/action-only clause that is always active. For an
    intent-bound clause, admission derives ``slot`` from the predicate's intent reference.
    Requiredness remains solely on the canonical :class:`IntentSlot`; ``compiled_expr``
    is admission-owned.
    """

    model_config = ConfigDict(extra="forbid")

    predicate: PredicateSpec
    slot: str | None = None
    compiled_expr: str = ""
    admission_status: GuardAdmissionStatus = GuardAdmissionStatus.PENDING
    admission_reason: str = ""


class GuardContract(BaseModel):
    """Typed, pre-compilation description of a transition's intended guard.

    A ``GuardContract`` is synthesis IR / metadata attached to a ``Transition``. It
    captures the guard's kind, the evidence used to classify the transition semantics,
    intent slots, independently admitted clauses, and admission bookkeeping. It does not
    by itself determine a runtime verdict; admission, clause activation, and DSL evaluation
    remain separate deterministic stages.
    """

    model_config = ConfigDict(extra="forbid")

    kind: GuardKind = GuardKind.UNKNOWN
    semantic_evidence: list[RequirednessEvidenceRef] = Field(default_factory=list)
    slots: list[IntentSlot] = Field(default_factory=list)
    clauses: list[GuardClause] = Field(default_factory=list)
    admission_status: GuardAdmissionStatus = GuardAdmissionStatus.PENDING
    admission_reason: str = ""


#: Human-readable admission_reason stamped on the placeholder below. Diagnostic
#: only -- ``is_runtime_pending_placeholder`` never reads it.
RUNTIME_PENDING_PLACEHOLDER_REASON = "runtime-witnessed edge awaiting semantic admission"


def runtime_pending_placeholder() -> GuardContract:
    """The contract stamped on a runtime-witnessed edge at creation.

    Deliberately empty: no clauses, no slots, kind UNKNOWN, ``admission_status``
    PENDING. Its whole job is to route re-encounters of the edge through the
    origin-agnostic present-but-pending rule (``classify_semantic_admission`` ->
    GUARD_INCONCLUSIVE -> UNCERTAIN) instead of the neutral guard, until the
    between-round admission pass discharges it by replacement. It is a
    dischargeable admission state, NOT a trust modifier. The same rule gates
    every pending contract regardless of origin.
    """
    return GuardContract(
        kind=GuardKind.UNKNOWN,
        admission_status=GuardAdmissionStatus.PENDING,
        admission_reason=RUNTIME_PENDING_PLACEHOLDER_REASON,
    )


def is_runtime_pending_placeholder(contract: GuardContract | None) -> bool:
    """Whether ``contract`` is an untouched runtime-edge placeholder.

    Structural test (PENDING, kind UNKNOWN, no clauses/slots/evidence) rather
    than a marker string, so a placeholder is recognized across serialization
    and never confused with a real analysis result: any contract that carries a
    clause, a slot, cited evidence, or a classified kind was produced by
    semantic analysis and is not a placeholder.
    """
    return (
        contract is not None
        and contract.admission_status is GuardAdmissionStatus.PENDING
        and contract.kind is GuardKind.UNKNOWN
        and not contract.clauses
        and not contract.slots
        and not contract.semantic_evidence
    )


class LlmGuardContractCandidate(BaseModel):
    """An LLM-produced guard-contract candidate, before admission.

    The LLM emits a typed transition :class:`GuardContract` (never free-form DSL) under
    provider structured output. This thin wrapper carries the parsed contract plus the
    model's self-reported completeness / rejection signal.

    ``parsed_ok`` is the authoritative success flag: ``True`` only when the provider returned
    a schema-valid object. When ``False`` (structured output unavailable, refusal, or
    validation failure) the pipeline must treat this as a clear rejection — it must NOT run
    admission on the empty placeholder ``contract`` or attach guard metadata as if admission
    succeeded.
    """

    contract: GuardContract = Field(default_factory=GuardContract)
    rejection_reason: str = ""
    parsed_ok: bool = False

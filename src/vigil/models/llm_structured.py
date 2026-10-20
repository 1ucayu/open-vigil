"""Lean, strict LLM-facing response models for guard/invariant generation.

The active generation path asks the LLM for the smallest typed object that can be
compiled and admitted deterministically. Audit/provenance/confidence/rejection reports are
owned by structured-output plumbing and symbolic admission, not by the model response.

Runtime models in :mod:`vigil.models.guard` and :mod:`vigil.models.invariant_candidate`
carry deterministic compilation/admission metadata. The LLM-facing models contain no
open dicts, free-form notes, self-rated confidence, semantic-completeness labels, or
effect-hint side channel.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vigil.models.guard import (
    GuardAdmissionStatus,
    GuardClause,
    GuardContract,
    GuardKind,
    GuardPredicateType,
    IntentSlot,
    LlmGuardContractCandidate,
    PredicateSpec,
    RequirednessEvidenceRef,
    RequirednessEvidenceSource,
    SlotType,
    ValueRef,
)
from vigil.models.invariant_candidate import (
    InvariantCandidatePacket,
    InvariantKind,
    StateInvariantCandidate,
)

InvariantPredicateTypeLiteral = Literal[
    "read",
    "value",
    "count",
]
# ``not_contains`` is deliberately absent: across every generation run it was never once
# proposed, its natural reading (assert absence) is unprovable under three-valued
# evaluation (missing element -> UNKNOWN, never TRUE), and soliciting it would violate the
# spec's own no-mechanical-operator-enumeration rule. The runtime DSL grammar/evaluator
# still accept it for hand-written specs; only the LLM-facing contract surface dropped it.
OperatorLiteral = Literal["==", "!=", ">", "<", ">=", "<=", "contains"]
InvariantOperatorLiteral = Literal["==", "contains"]
ReadPropertyLiteral = Literal["text", "content_description", "is_checked", "is_selected"]
ActionPropertyLiteral = Literal["target_text", "target_content_desc", "input_text"]


class StrictLiteralValueRef(BaseModel):
    """RHS value reference for invariant predicates: literals only, no intent binding."""

    model_config = ConfigDict(extra="forbid")

    value: str | int | float | bool

    def to_runtime(self) -> ValueRef:
        return ValueRef(kind="literal", value=self.value)


class StrictGuardValueRef(BaseModel):
    """Provider-safe RHS union for a literal or one frozen intent slot.

    The two alternatives share one object schema because OpenAI strict structured output
    rejects JSON Schema ``oneOf``.  The discriminator remains semantic: deterministic
    validation enforces exactly one populated payload before conversion to ``ValueRef``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["literal", "intent"]
    value: str | int | float | bool | None
    slot: str | None

    @model_validator(mode="after")
    def validate_reference_shape(self) -> StrictGuardValueRef:
        if self.kind == "literal":
            if self.value is None or self.slot is not None:
                raise ValueError("literal references require value and slot=null")
        elif self.value is not None or not self.slot:
            raise ValueError("intent references require value=null and a non-empty slot")
        return self

    def to_runtime(self) -> ValueRef:
        if self.kind == "literal":
            return ValueRef(kind="literal", value=self.value)
        return ValueRef(kind="intent", slot=self.slot)


class StrictPredicateSpec(BaseModel):
    """One typed predicate. The runtime ``args`` open dict is intentionally absent."""

    model_config = ConfigDict(extra="forbid")

    predicate_type: GuardPredicateType
    element: str = ""
    property: ReadPropertyLiteral | ActionPropertyLiteral | Literal[""] = ""
    operator: OperatorLiteral
    expected: StrictGuardValueRef

    @model_validator(mode="after")
    def validate_predicate_shape(self) -> StrictPredicateSpec:
        if self.predicate_type == "read":
            if not self.element or self.property not in {
                "text",
                "content_description",
                "is_checked",
                "is_selected",
            }:
                raise ValueError("read predicates require an element and readable property")
        elif self.predicate_type in {"value", "count"}:
            if not self.element or self.property:
                raise ValueError(
                    f"{self.predicate_type} predicates require an element and no property"
                )
        elif self.predicate_type == "action" and (
            self.element
            or self.property
            not in {
                "target_text",
                "target_content_desc",
                "input_text",
            }
        ):
            raise ValueError("action predicates require one semantic action property")
        return self

    def to_runtime(self) -> PredicateSpec:
        return PredicateSpec(
            predicate_type=self.predicate_type,
            element=self.element,
            property=self.property,
            operator=self.operator,
            expected=self.expected.to_runtime(),
        )


class StrictInvariantPredicateSpec(BaseModel):
    """One state-invariant predicate. Scope is current-state widgets plus literals only."""

    model_config = ConfigDict(extra="forbid")

    predicate_type: InvariantPredicateTypeLiteral
    element: str = ""
    property: ReadPropertyLiteral | Literal[""] = ""
    operator: InvariantOperatorLiteral
    expected: StrictLiteralValueRef

    @model_validator(mode="after")
    def validate_predicate_shape(self) -> StrictInvariantPredicateSpec:
        if self.predicate_type == "read":
            if not self.element or self.property not in {
                "text",
                "content_description",
                "is_checked",
                "is_selected",
            }:
                raise ValueError("read predicates require an element and readable property")
        elif not self.element or self.property:
            raise ValueError(f"{self.predicate_type} predicates require an element and no property")
        return self

    def to_runtime(self) -> PredicateSpec:
        return PredicateSpec(
            predicate_type=self.predicate_type,
            element=self.element,
            property=self.property,
            operator=self.operator,
            expected=self.expected.to_runtime(),
        )


class StrictRequirednessEvidenceRef(BaseModel):
    """One LLM citation into the concrete transition evidence packet."""

    model_config = ConfigDict(extra="forbid")

    source: RequirednessEvidenceSource
    identifier: str

    def to_runtime(self) -> RequirednessEvidenceRef:
        return RequirednessEvidenceRef(source=self.source, identifier=self.identifier)


class StrictIntentSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    slot_type: SlotType
    description: str
    value_domain: list[str]

    @model_validator(mode="after")
    def validate_value_domain_scope(self) -> StrictIntentSlot:
        if self.slot_type == "enum" and not self.value_domain:
            raise ValueError("enum slots require a non-empty value_domain")
        if self.slot_type != "enum" and self.value_domain:
            raise ValueError("value_domain is only valid for enum slots")
        return self

    def to_runtime(self) -> IntentSlot:
        return IntentSlot(
            name=self.name,
            slot_type=self.slot_type,
            description=self.description,
            value_domain=list(self.value_domain),
        )


class StrictGuardClause(BaseModel):
    """One independently activatable predicate emitted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    predicate: StrictPredicateSpec

    def to_runtime(self) -> GuardClause:
        predicate = self.predicate.to_runtime()
        expected = self.predicate.expected
        slot = expected.slot if expected.kind == "intent" else None
        return GuardClause(
            predicate=predicate,
            slot=slot,
            admission_status=GuardAdmissionStatus.PENDING,
        )


class StrictGuardContract(BaseModel):
    """Minimal guard contract shape emitted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    kind: GuardKind
    semantic_evidence: list[StrictRequirednessEvidenceRef] = Field(default_factory=list)
    slots: list[StrictIntentSlot]
    clauses: list[StrictGuardClause]

    def to_runtime(self) -> GuardContract:
        return GuardContract(
            kind=self.kind,
            semantic_evidence=[ref.to_runtime() for ref in self.semantic_evidence],
            slots=[slot.to_runtime() for slot in self.slots],
            clauses=[clause.to_runtime() for clause in self.clauses],
            admission_status=GuardAdmissionStatus.PENDING,
            admission_reason="",
        )


class LlmTransitionGuardResponse(BaseModel):
    """Canonical guard response: exactly one minimal contract."""

    model_config = ConfigDict(extra="forbid")

    contract: StrictGuardContract

    def to_runtime(self) -> LlmGuardContractCandidate:
        contract = self.contract.to_runtime()
        return LlmGuardContractCandidate(
            contract=contract,
            rejection_reason="",
        )


class StrictStateInvariantSpec(BaseModel):
    """One classified, typed invariant predicate proposal."""

    model_config = ConfigDict(extra="forbid")

    kind: InvariantKind
    predicate: StrictInvariantPredicateSpec

    def to_runtime(self) -> StateInvariantCandidate:
        return StateInvariantCandidate(
            kind=self.kind,
            predicate=self.predicate.to_runtime(),
            source="llm",
        )


class LlmInvariantGuardResponse(BaseModel):
    """Canonical invariant response: minimal state-invariant candidates only."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[StrictStateInvariantSpec]

    def to_runtime(self) -> InvariantCandidatePacket:
        return InvariantCandidatePacket(
            candidates=[candidate.to_runtime() for candidate in self.candidates]
        )

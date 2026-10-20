"""Deterministic admission for typed state-invariant candidates.

Admission validates one current-state predicate, lowers its widget alias to a stable
runtime resource id, compiles it once, and requires it to evaluate true on every runtime
observation assigned to the state. Invariants have no intent slots and are always active.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from vigil.models.fsm import StateInvariant
from vigil.models.guard import PredicateSpec
from vigil.neuro.guard_admission import resolve_element
from vigil.neuro.guard_dsl_compiler import compile_predicate_spec
from vigil.neuro.guard_registry import WidgetRole
from vigil.symbolic.dsl_evaluator import DSLEvaluator, GuardStatus
from vigil.symbolic.screen_context import (
    runtime_elements_from_observation,
    screen_context_from_observation,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vigil.models.invariant_candidate import StateInvariantCandidate
    from vigil.neuro.invariant_evidence import InvariantEvidence

_TEXT_PROPERTIES = frozenset({"text", "content_description"})
_BOOLEAN_PROPERTIES = frozenset({"is_checked", "is_selected"})
_TEXT_OPERATORS = frozenset({"==", "contains"})


class InvariantAdmissionResult(BaseModel):
    """Outcome of admitting one typed invariant candidate."""

    admitted: bool
    invariant: StateInvariant | None = None
    reason: str = ""
    compiled_expr: str = ""


def _reject(reason: str) -> InvariantAdmissionResult:
    return InvariantAdmissionResult(admitted=False, reason=reason)


def _duplicate_resource_observations(
    observations: list[dict[str, Any]], resource_id: str
) -> list[str]:
    duplicates: list[str] = []
    for observation in observations:
        count = sum(
            str(element.get("resource_id") or "") == resource_id
            for element in runtime_elements_from_observation(observation)
        )
        if count > 1:
            duplicates.append(str(observation.get("screen_id") or "<unknown>"))
    return duplicates


def _predicate_error(predicate: PredicateSpec, role: WidgetRole | None) -> str:
    expected = predicate.expected
    if expected is None or expected.kind != "literal":
        return "state invariants require a literal expected value"
    value = expected.value
    operator = predicate.operator

    if predicate.predicate_type == "read":
        if predicate.property in _TEXT_PROPERTIES:
            if operator not in _TEXT_OPERATORS:
                return f"operator {operator!r} is not valid for invariant text"
            if not isinstance(value, str):
                return "text invariant literals must be strings"
            return ""
        if predicate.property in _BOOLEAN_PROPERTIES:
            if operator != "==":
                return f"operator {operator!r} is not valid for invariant boolean state"
            if not isinstance(value, bool):
                return "boolean invariant literals must be JSON booleans"
            return ""
        return f"property {predicate.property!r} is outside invariant vocabulary"

    if predicate.predicate_type == "value":
        if operator not in _TEXT_OPERATORS:
            return f"operator {operator!r} is not valid for invariant text"
        if not isinstance(value, str):
            return "value invariant literals must be strings"
        return ""

    if predicate.predicate_type == "count":
        if role is not WidgetRole.LIST_CONTAINER:
            return "count invariant requires an observed list/container role"
        if operator != "==":
            return "count invariants support exact equality only"
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return "count invariant literals must be non-negative integers"
        return ""

    return f"predicate type {predicate.predicate_type!r} is outside invariant vocabulary"


def _kind_error(candidate: StateInvariantCandidate) -> str:
    """Require the semantic kind to agree with the predicate's observable type."""
    predicate = candidate.predicate
    kind = candidate.kind
    if kind == "unknown":
        return ""
    if kind == "container_shape":
        return "" if predicate.predicate_type == "count" else "container_shape requires count"
    if kind == "semantic_role":
        if predicate.predicate_type == "read" and predicate.property in _BOOLEAN_PROPERTIES:
            return ""
        return "semantic_role requires a checked/selected boolean predicate"
    # stable_label / form_status / status
    if predicate.predicate_type == "value":
        return ""
    if predicate.predicate_type == "read" and predicate.property in _TEXT_PROPERTIES:
        return ""
    return f"{kind} requires a text/value predicate"


@lru_cache(maxsize=1)
def _evaluator() -> DSLEvaluator:
    return DSLEvaluator()


def admit_state_invariant_candidate(
    candidate: StateInvariantCandidate,
    evidence: InvariantEvidence,
    *,
    evaluator: DSLEvaluator | None = None,
) -> InvariantAdmissionResult:
    """Admit one typed predicate after exact replay over all state observations."""
    predicate = candidate.predicate
    if not predicate.element:
        return _reject(f"{predicate.predicate_type} predicate requires an element")

    resource_id, entry, resolution_error = resolve_element(
        predicate.element, evidence.arrival_registry
    )
    if resource_id is None:
        return _reject(f"element not runtime-resolvable ({resolution_error})")

    error = _predicate_error(predicate, entry.role if entry is not None else None)
    if error:
        return _reject(error)
    expected = predicate.expected
    if (
        expected is not None
        and isinstance(expected.value, str)
        and expected.value in evidence.exploration_payload_values
    ):
        return _reject("literal is an input payload introduced by offline exploration provenance")

    kind_error = _kind_error(candidate)
    if kind_error:
        return _reject(kind_error)

    lowered = predicate.model_copy(update={"element": resource_id})
    compiled = compile_predicate_spec(lowered)
    if compiled is None:
        return _reject("typed predicate could not be compiled")

    observations = evidence.observations
    if not observations:
        return _reject("no observation is available for invariant admission")
    observed_ids = {str(observation.get("screen_id") or "") for observation in observations}
    missing_ids = sorted(set(evidence.raw_screen_ids) - observed_ids)
    if missing_ids:
        return _reject(f"cited state observations are unavailable: {missing_ids}")
    duplicates = _duplicate_resource_observations(observations, resource_id)
    if duplicates:
        return _reject(f"resource_id {resource_id!r} is ambiguous in observations {duplicates}")

    runtime_evaluator = evaluator or _evaluator()
    for observation in observations:
        result = runtime_evaluator.evaluate(
            compiled,
            screen_ctx=screen_context_from_observation(observation),
        )
        if result.status is not GuardStatus.TRUE:
            return _reject(
                f"not true in observation {observation.get('screen_id') or '<unknown>'}: "
                f"{result.status.value} — {result.failure_reason or 'condition not satisfied'}"
            )

    invariant = StateInvariant(
        expr=compiled,
        source=candidate.source,
        evidence_count=len(observations),
    )
    return InvariantAdmissionResult(
        admitted=True,
        invariant=invariant,
        reason=f"admitted: true across {len(observations)} observations",
        compiled_expr=compiled,
    )

"""Admission validation for compiled guard contracts (step 4) — the executability gate.

Admission decides whether a synthesized :class:`~vigil.models.guard.GuardContract` may be
attached to a transition as an executable guard. Unlike a pure grammar check, admission
guarantees the resulting DSL is **runtime-executable by the current**
:class:`~vigil.symbolic.decision_engine.DecisionEngine` context:

- Element references are lowered to a stable, offline-known, runtime-resolvable key — the
  full ``resource_id``. Registry-only slug aliases and synthesized ``Class_idx`` aliases
  (whose live index is not knowable offline) are rejected rather than guessed.
- Action properties are restricted to the semantic surfaces in the generation contract:
  ``target_text``, ``target_content_desc``, and ``input_text``.
- ``$intent.*`` slots must be declared in ``contract.slots``.
- Exploration input payloads are identified from transition provenance rather than by
  matching particular strings.
- Every intent-bound clause must link to one admitted required/optional slot. A clause
  without an intent slot is admitted on its own evidence and is always active at runtime.

Anything that cannot be guaranteed executable is ``REJECTED`` and remains an offline
audit candidate; only admitted contracts are attached to the EFSM.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from lark import Lark
from lark.exceptions import LarkError
from pydantic import BaseModel, Field

from vigil.core.paths import resolve_dsl_grammar_path
from vigil.models.guard import (
    CounterfactualOutcome,
    GuardAdmissionStatus,
    GuardClause,
    GuardContract,
    GuardKind,
    IntentSlot,
    PredicateSpec,
    RequirednessEvidence,
    RequirednessEvidenceRef,
    SlotRequiredness,
    SlotType,
    ValueRef,
)
from vigil.neuro.guard_dsl_compiler import compile_predicate_spec
from vigil.neuro.guard_evidence import evidence_reference_catalog
from vigil.neuro.guard_registry import WidgetRole
from vigil.symbolic.dsl_evaluator import (
    DSLEvaluator,
    GuardStatus,
    normalize_text_for_compare,
)
from vigil.symbolic.numeric_normalizer import (
    NumericNormalizationError,
    normalize_numeric_scalar,
)
from vigil.symbolic.screen_context import (
    runtime_elements_from_observation,
    screen_context_from_observation,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vigil.neuro.guard_evidence import GuardEvidence
    from vigil.neuro.guard_registry import WidgetRegistry, WidgetRegistryEntry


# Canonical generation vocabulary. ``value`` and ``count`` have dedicated predicate
# forms, so their equivalent ``read(..., value|children_count)`` spellings are excluded.
_READABLE_PROPS: frozenset[str] = frozenset(
    {"text", "content_description", "is_checked", "is_selected"}
)
_READABLE_BOOL_PROPS: frozenset[str] = frozenset({"is_checked", "is_selected"})
_READABLE_STRING_PROPS: frozenset[str] = frozenset({"text", "content_description"})

# Action properties the runtime action context exposes (decision_engine._build_action_context).
_ALLOWED_ACTION_PROPS: frozenset[str] = frozenset(
    {"target_text", "target_content_desc", "input_text"}
)
_CONTAINMENT_OPS: frozenset[str] = frozenset({"contains"})
_ORDERED_OPS: frozenset[str] = frozenset({">", "<", ">=", "<="})
_TEXTUAL_OPS: frozenset[str] = frozenset({"==", "!=", "contains"})
_BOOLEAN_OPS: frozenset[str] = frozenset({"==", "!="})
_NUMERIC_OPS: frozenset[str] = frozenset({"==", "!=", ">", "<", ">=", "<="})


def _literal_type_error(ptype: str, prop: str, value: object) -> str:
    """Return an admission rejection reason for literal/property type mismatch."""
    if ptype == "read":
        if prop in _READABLE_BOOL_PROPS and not isinstance(value, bool):
            return f"literal for boolean property '{prop}' must be JSON boolean"
        if prop in _READABLE_STRING_PROPS and not isinstance(value, str):
            return f"literal for string property '{prop}' must be JSON string"
    if ptype == "action" and prop in _ALLOWED_ACTION_PROPS and not isinstance(value, str):
        return f"literal for action property '{prop}' must be JSON string"
    if ptype == "count" and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        return "literal for count predicate must be a non-negative JSON integer"
    return ""


class GuardAdmissionResult(BaseModel):
    """Outcome of admitting a guard contract."""

    admitted: bool
    status: GuardAdmissionStatus
    guard: str | None = None
    reason: str = ""
    rejected_predicates: list[str] = Field(default_factory=list)
    clauses: list[GuardClause] = Field(default_factory=list)
    slots: list[IntentSlot] | None = None


def apply_guard_admission(
    contract: GuardContract,
    result: GuardAdmissionResult,
) -> GuardContract:
    """Materialize the single runtime contract produced by admission.

    Admission reports retain every candidate clause for audit.  Runtime metadata keeps
    only independently admitted executable clauses, so offline generation and
    micro-evolution share exactly one attachment rule.
    """
    contract.admission_status = result.status
    contract.admission_reason = result.reason
    if result.slots is not None:
        contract.slots = list(result.slots)
    contract.clauses = (
        [
            clause
            for clause in result.clauses
            if clause.admission_status is GuardAdmissionStatus.ADMITTED and clause.compiled_expr
        ]
        if result.admitted
        else []
    )
    return contract


@lru_cache(maxsize=1)
def _parser() -> Lark:
    grammar_text = resolve_dsl_grammar_path().read_text(encoding="utf-8")
    return Lark(grammar_text, parser="earley", start="start", keep_all_tokens=True)


@lru_cache(maxsize=1)
def _evaluator() -> DSLEvaluator:
    return DSLEvaluator()


@dataclass(frozen=True)
class _Lowered:
    """Result of lowering one predicate: a compiled string, or a rejection reason."""

    compiled: str | None
    reason: str
    resource_id: str | None = None


def resolve_element(
    element: str, registry: WidgetRegistry
) -> tuple[str | None, WidgetRegistryEntry | None, str]:
    """Lower an element reference to a runtime-resolvable key (full ``resource_id``)."""
    entry = registry.entries.get(element)
    if entry is not None:
        if entry.resource_id:
            if registry.resource_id_to_alias.get(entry.resource_id) != entry.alias:
                return (
                    None,
                    None,
                    f"element '{element}' has ambiguous resource_id {entry.resource_id!r}",
                )
            return entry.resource_id, entry, ""
        return (
            None,
            None,
            f"element '{element}' has no resource_id (no runtime-resolvable key)",
        )
    if element in registry.resource_id_to_alias:
        # Already a known resource_id — use verbatim.
        alias = registry.resource_id_to_alias[element]
        return element, registry.entries.get(alias), ""
    return None, None, f"element '{element}' not present in source registry"


def _predicate_source_type(pred: PredicateSpec) -> str:
    """Return the runtime value category exposed by a predicate's left side."""
    if pred.predicate_type == "count":
        return "count"
    if pred.predicate_type == "read":
        if pred.property in _READABLE_BOOL_PROPS:
            return "boolean"
        if pred.property in _READABLE_STRING_PROPS:
            return "text"
    if pred.predicate_type in {"value", "action"}:
        return "text"
    return "unknown"


def _intent_predicate_error(pred: PredicateSpec, slot: IntentSlot) -> str:
    """Validate one intent predicate against the centralized type/operator matrix."""
    op = pred.operator or "=="
    source_type = _predicate_source_type(pred)
    if slot.slot_type is SlotType.NUMBER:
        if pred.operator is None:
            return f"numeric intent predicate for slot '{slot.name}' requires an explicit operator"
        if op not in _NUMERIC_OPS:
            return f"operator '{op}' is not valid for number slot '{slot.name}'"
        if pred.predicate_type == "action" and pred.property != "input_text":
            return (
                f"number slot '{slot.name}' may bind only action(input_text), "
                f"not action({pred.property})"
            )
        if source_type == "boolean":
            return (
                f"number slot '{slot.name}' cannot bind boolean source property '{pred.property}'"
            )
        return ""
    if slot.slot_type is SlotType.BOOLEAN:
        if op not in _BOOLEAN_OPS:
            return f"operator '{op}' is not valid for boolean slot '{slot.name}'"
        if source_type != "boolean":
            return (
                f"boolean slot '{slot.name}' requires a boolean source property, not {source_type}"
            )
        return ""
    if slot.slot_type in {SlotType.STRING, SlotType.ENUM}:
        if op not in _TEXTUAL_OPS:
            return f"operator '{op}' is not valid for {slot.slot_type.value} slot '{slot.name}'"
        if source_type in {"boolean", "count"}:
            return (
                f"{slot.slot_type.value} slot '{slot.name}' cannot bind {source_type} "
                "source property"
            )
    return ""


def _literal_operator_error(pred: PredicateSpec, expected: ValueRef | None) -> str:
    """Apply the same source/operator matrix to literal-only clauses."""
    if expected is None or expected.kind != "literal":
        return ""
    if expected.value is None:
        return "literal value cannot be null"
    op = pred.operator or "=="
    source_type = _predicate_source_type(pred)
    if source_type == "boolean" and op not in _BOOLEAN_OPS:
        return f"operator '{op}' is not valid for boolean source property '{pred.property}'"
    if source_type == "count" and op not in _NUMERIC_OPS:
        return f"operator '{op}' is not valid for count source"
    if source_type == "text" and not isinstance(expected.value, str):
        if isinstance(expected.value, bool):
            return "boolean literal requires a boolean source property"
        return "text/value literal predicates require a JSON string"
    if source_type == "text" and op in _ORDERED_OPS:
        return "ordered comparison on text requires a number intent slot"
    if source_type == "text" and op not in _TEXTUAL_OPS:
        return f"operator '{op}' is not valid for a string value"
    return ""


def _lower_predicate(
    pred: PredicateSpec, registry: WidgetRegistry, declared_slots: dict[str, IntentSlot]
) -> _Lowered:
    ptype = pred.predicate_type
    expected = pred.expected
    # Intent slot must be declared in the contract.
    if (
        expected is not None
        and expected.kind == "intent"
        and (not expected.slot or expected.slot not in declared_slots)
    ):
        return _Lowered(None, f"undeclared intent slot '{expected.slot}'")

    intent_slot = (
        declared_slots.get(expected.slot or "")
        if expected is not None and expected.kind == "intent"
        else None
    )
    if intent_slot is not None:
        type_error = _intent_predicate_error(pred, intent_slot)
        if type_error:
            return _Lowered(None, type_error)
    literal_operator_error = _literal_operator_error(pred, expected)
    if literal_operator_error:
        return _Lowered(None, literal_operator_error)

    lowered = pred
    resource_id: str | None = None

    if ptype in ("read", "value", "count"):
        if not pred.element:
            return _Lowered(None, f"{ptype} predicate missing element")
        key, entry, reason = resolve_element(pred.element, registry)
        if key is None:
            return _Lowered(None, reason)
        if ptype == "count" and (entry is None or entry.role is not WidgetRole.LIST_CONTAINER):
            return _Lowered(None, "count predicate requires an observed list/container role")
        resource_id = key
        op = pred.operator or "=="
        # Unsatisfiable-as-authored: value(X) under a scalar operator reads X's OWN
        # text/value, but for an unlabeled container (e.g. a Compose FilterChip whose
        # label is a child node) that scalar is empty in every source observation —
        # the visible value lives in descendant_texts. Equality then compares "" to
        # the expected value and can only be FALSE at runtime (observed: 41 false
        # DENYs on the clock timer.start / duration-chip guards, 2026-08-07 arm).
        # Containment is the operator defined over descendant values; demand it.
        if (
            ptype == "value"
            and op not in _CONTAINMENT_OPS
            and entry is not None
            and not entry.text.strip()
            and not str(entry.observed_values.get("value") or "").strip()
            and entry.descendant_texts
        ):
            return _Lowered(
                None,
                (
                    "scalar operator over container with empty own value: visible "
                    f"value of '{pred.element}' lives on descendants "
                    f"{entry.descendant_texts!r}; use a containment operator or a "
                    "labeled child element alias"
                ),
            )
        if ptype == "count" and expected is not None and expected.kind == "literal":
            if op in _CONTAINMENT_OPS:
                return _Lowered(None, "containment operator is not valid for count(...)")
            type_error = _literal_type_error(ptype, "", expected.value)
            if type_error:
                return _Lowered(None, type_error)
        if ptype == "read":
            if not pred.property or pred.property not in _READABLE_PROPS:
                return _Lowered(None, f"property '{pred.property}' not runtime-readable")
            if op in _CONTAINMENT_OPS and pred.property not in _READABLE_STRING_PROPS:
                return _Lowered(
                    None,
                    f"containment operator is not valid for non-string property '{pred.property}'",
                )
            if expected is not None and expected.kind == "literal":
                type_error = _literal_type_error(ptype, pred.property, expected.value)
                if type_error:
                    return _Lowered(None, type_error)
        lowered = pred.model_copy(update={"element": key})

    elif ptype == "action":
        prop = pred.property or ""
        if prop not in _ALLOWED_ACTION_PROPS:
            return _Lowered(None, f"action property '{pred.property}' not runtime-resolvable")
        if expected is not None and expected.kind == "literal":
            type_error = _literal_type_error(ptype, prop, expected.value)
            if type_error:
                return _Lowered(None, type_error)
        lowered = pred.model_copy(update={"property": prop})

    compiled = compile_predicate_spec(lowered)
    if compiled is None:
        return _Lowered(None, f"{ptype} predicate not compilable")
    return _Lowered(compiled, "", resource_id)


def _action_type(action: dict[str, object]) -> str:
    return str(action.get("type") or action.get("action_type") or "").strip().lower()


def _is_canonical_action_target_binding(predicate: PredicateSpec) -> bool:
    """Return whether an intent equality merely repeats the action's target identity."""
    expected = predicate.expected
    return (
        predicate.predicate_type == "action"
        and predicate.property in {"target_text", "target_content_desc"}
        and (predicate.operator or "==") == "=="
        and expected is not None
        and expected.kind == "intent"
    )


def _action_semantic_value(
    action: dict[str, object],
    property_name: str,
    registry: WidgetRegistry,
) -> str:
    direct = action.get(property_name)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    selector = action.get("target_selector")
    if isinstance(selector, dict):
        selector_key = "text" if property_name == "target_text" else "content_description"
        selected = selector.get(selector_key)
        if isinstance(selected, str) and selected.strip():
            return selected.strip()

    target = str(action.get("target") or "")
    alias = registry.element_id_to_alias.get(target)
    if alias is None:
        resource_id = str(action.get("target_resource_id") or action.get("resource_id") or "")
        alias = registry.resource_id_to_alias.get(resource_id)
    entry = registry.entries.get(alias or "")
    if entry is None:
        return ""
    value = entry.text if property_name == "target_text" else entry.content_description
    return value.strip()


def _requiredness_refs(
    evidence: GuardEvidence,
    *,
    sibling_indexes: list[int] | None = None,
    source_element: str | None = None,
    effect_id: str | None = None,
) -> list[RequirednessEvidenceRef]:
    refs = [RequirednessEvidenceRef(source="known_action", identifier="type")]
    if source_element:
        source_alias = (
            source_element
            if source_element in evidence.source_registry.entries
            else evidence.source_registry.resource_id_to_alias.get(source_element)
        )
        if source_alias:
            refs.append(RequirednessEvidenceRef(source="source_widget", identifier=source_alias))
    for index in sibling_indexes or []:
        refs.append(RequirednessEvidenceRef(source="sibling_action", identifier=str(index)))
    if effect_id:
        refs.append(RequirednessEvidenceRef(source="effect_metadata", identifier=effect_id))
    catalog = {
        (catalog_ref.source, catalog_ref.identifier)
        for catalog_ref in evidence_reference_catalog(evidence)
    }
    return [ref for ref in refs if (ref.source, ref.identifier) in catalog]


def _semantic_evidence_error(
    contract: GuardContract,
    evidence: GuardEvidence,
    *,
    require_complete: bool,
) -> str:
    """Validate the evidence certificate for transition semantic classification.

    Executable clauses themselves expose the semantic relation they enforce.  An empty
    ``none``/``navigation`` conjunction has no such witness, so it additionally must cite
    every packet surface that deterministic admission knows is relevant to classification.
    The LLM still performs the semantic classification; admission only verifies that its
    certificate is grounded in this transition's evidence packet.
    """
    keys = {(ref.source, ref.identifier) for ref in contract.semantic_evidence}
    catalog = {(ref.source, ref.identifier) for ref in evidence_reference_catalog(evidence)}
    invalid = [f"{source}:{identifier}" for source, identifier in sorted(keys - catalog)]
    if invalid:
        return "transition semantic evidence has unresolved reference(s): " + ", ".join(invalid)
    if not require_complete:
        return ""

    required = {("known_action", "type")}
    if evidence.action_target_alias is not None:
        required.add(("source_widget", evidence.action_target_alias))
    if evidence.diff_summary:
        required.add(("source_target_diff", "source_to_target"))
    required.update(("effect_metadata", effect.effect_id) for effect in evidence.effect_metadata)
    missing = sorted(required - keys)
    if missing:
        rendered = ", ".join(f"{source}:{identifier}" for source, identifier in missing)
        return "empty semantic conjunction lacks classification evidence: " + rendered
    return ""


def _r3_semantic_content(
    action: dict[str, object],
    property_name: str,
    registry: WidgetRegistry,
) -> str:
    """Semantic content of an action's own control for R3 sibling comparison.

    Falls back from the control's direct text/content_description to its
    registry ``descendant_texts`` (unlabeled containers such as duration chips
    carry their visible value on a child).
    """
    value = _action_semantic_value(action, property_name, registry)
    if value:
        return value
    target = str(action.get("target") or "")
    alias = registry.element_id_to_alias.get(target)
    if alias is None:
        resource_id = str(action.get("target_resource_id") or action.get("resource_id") or "")
        alias = registry.resource_id_to_alias.get(resource_id)
    entry = registry.entries.get(alias or "")
    if entry is not None and entry.descendant_texts:
        return "|".join(entry.descendant_texts)
    return ""


def _ancestor_ids(element_id: str, elements_by_id: dict[str, dict[str, object]]) -> list[str]:
    """Parent-chain element ids from the element upward (excluding itself)."""
    chain: list[str] = []
    seen: set[str] = set()
    current = elements_by_id.get(element_id)
    while current is not None:
        parent_id = str(current.get("parent_id") or "")
        if not parent_id or parent_id in seen:
            break
        seen.add(parent_id)
        chain.append(parent_id)
        current = elements_by_id.get(parent_id)
    return chain


def _find_action_element(
    observation_elements: list[dict[str, object]],
    elements_by_id: dict[str, dict[str, object]],
    action: dict[str, object],
) -> dict[str, object] | None:
    """Locate one action's concrete target element in a source observation."""
    target_id = str(action.get("target") or "")
    if target_id and target_id in elements_by_id:
        return elements_by_id[target_id]
    resource_id = str(action.get("target_resource_id") or action.get("resource_id") or "")
    if not resource_id:
        return None
    matches = [
        element
        for element in observation_elements
        if str(element.get("resource_id") or "") == resource_id
    ]
    return matches[0] if len(matches) == 1 else None


def _choice_row(
    target_element_id: str,
    elements_by_id: dict[str, dict[str, object]],
    forbidden_ids: set[str],
) -> str:
    """Highest ancestor of the target whose subtree holds no forbidden target.

    This is the structural "choice row": the widest container that still
    discriminates this action from its same-type siblings. Falls back to the
    target itself when even its direct parent already contains a sibling.
    """
    forbidden_closures = [
        {forbidden_id, *_ancestor_ids(forbidden_id, elements_by_id)}
        for forbidden_id in forbidden_ids
    ]
    row = target_element_id
    for ancestor_id in _ancestor_ids(target_element_id, elements_by_id):
        if any(ancestor_id in closure for closure in forbidden_closures):
            break
        row = ancestor_id
    return row


def _row_content(
    row_id: str,
    observation_elements: list[dict[str, object]],
    elements_by_id: dict[str, dict[str, object]],
) -> str:
    """Readable text content of one choice row's subtree (order-stable)."""
    texts: list[str] = []
    for element in observation_elements:
        element_id = str(element.get("element_id") or "")
        if not element_id:
            continue
        if element_id != row_id and row_id not in _ancestor_ids(element_id, elements_by_id):
            continue
        value = str(element.get("text") or element.get("content_description") or "").strip()
        if value and value not in texts:
            texts.append(value)
    return "|".join(texts)


def _classify_clause_requiredness(
    predicate: PredicateSpec,
    slot: IntentSlot,
    evidence: GuardEvidence,
) -> RequirednessEvidence:
    """Recompute one clause's counterfactual class from machine-checkable evidence."""
    operator = predicate.operator or "=="
    action_type = _action_type(evidence.action)

    if slot.slot_type is SlotType.NUMBER and operator in _ORDERED_OPS:
        return RequirednessEvidence(
            outcome=CounterfactualOutcome.ACTION_REMAINS_VALID,
            refs=_requiredness_refs(evidence, source_element=predicate.element),
        )
    if operator == "!=" and not (
        slot.slot_type is SlotType.BOOLEAN
        or (predicate.predicate_type == "read" and predicate.property in _READABLE_BOOL_PROPS)
    ):
        return RequirednessEvidence(
            outcome=CounterfactualOutcome.ACTION_REMAINS_VALID,
            refs=_requiredness_refs(evidence, source_element=predicate.element),
        )
    if (
        predicate.predicate_type == "action"
        and predicate.property == "input_text"
        and operator == "=="
        and action_type == "input_text"
    ):
        return RequirednessEvidence(
            outcome=CounterfactualOutcome.PAYLOAD_UNDERDETERMINED,
            refs=_requiredness_refs(evidence, effect_id="action_value_payload"),
        )
    if (
        predicate.predicate_type == "action"
        and predicate.property in {"target_text", "target_content_desc"}
        and operator == "=="
        and action_type != "input_text"
    ):
        current = _action_semantic_value(
            evidence.action,
            predicate.property,
            evidence.source_registry,
        )
        siblings = [
            index
            for index, sibling in enumerate(evidence.sibling_actions)
            if _action_type(sibling) == action_type
            and (
                sibling_value := _action_semantic_value(
                    sibling,
                    predicate.property,
                    evidence.source_registry,
                )
            )
            and current
            and sibling_value != current
        ]
        if siblings:
            return RequirednessEvidence(
                outcome=CounterfactualOutcome.TARGET_UNDERDETERMINED,
                refs=_requiredness_refs(
                    evidence,
                    sibling_indexes=siblings,
                    effect_id="resolved_action_target",
                ),
            )
    # R3: a screen-read equality over the action's own target (R3a) or over an
    # element inside the action's discriminating semantic surface (R3b) is a
    # choice binding whenever same-type siblings present different non-empty
    # content on their corresponding surface — the slot selects which
    # row/control is acted on.
    if (
        predicate.predicate_type in {"read", "value"}
        and operator in {"==", "contains"}
        and predicate.expected is not None
        and predicate.expected.kind == "intent"
        and action_type != "input_text"
        and predicate.element
    ):
        element_alias = (
            predicate.element
            if predicate.element in evidence.source_registry.entries
            else evidence.source_registry.resource_id_to_alias.get(predicate.element)
        )
        comparison_property = (
            "target_content_desc"
            if predicate.property in {"content_desc", "content_description"}
            else "target_text"
        )
        # R3a: the clause reads the selected control itself.
        if element_alias is not None and element_alias == evidence.action_target_alias:
            current = _r3_semantic_content(
                evidence.action, comparison_property, evidence.source_registry
            )
            siblings = [
                index
                for index, sibling in enumerate(evidence.sibling_actions)
                if _action_type(sibling) == action_type
                and (
                    sibling_value := _r3_semantic_content(
                        sibling, comparison_property, evidence.source_registry
                    )
                )
                and current
                and sibling_value != current
            ]
            if siblings:
                return RequirednessEvidence(
                    outcome=CounterfactualOutcome.TARGET_UNDERDETERMINED,
                    refs=_requiredness_refs(
                        evidence,
                        sibling_indexes=siblings,
                        source_element=predicate.element,
                        effect_id="resolved_action_target",
                    ),
                )
        # R3c: the clause reads inside the action's structural "choice row" —
        # the widest ancestor containing this target but no same-type sibling
        # target — and sibling choice rows carry different non-empty content.
        element_entry = (
            evidence.source_registry.entries.get(element_alias)
            if element_alias is not None
            else None
        )
        if element_entry is not None:
            for observation in evidence.source_observations:
                observation_elements = runtime_elements_from_observation(observation)
                elements_by_id = {
                    str(element.get("element_id") or ""): element
                    for element in observation_elements
                    if element.get("element_id")
                }
                my_element = _find_action_element(
                    observation_elements, elements_by_id, evidence.action
                )
                if my_element is None:
                    continue
                my_id = str(my_element.get("element_id") or "")
                sibling_elements: dict[int, str] = {}
                for index, sibling in enumerate(evidence.sibling_actions):
                    if _action_type(sibling) != action_type:
                        continue
                    found = _find_action_element(observation_elements, elements_by_id, sibling)
                    if found is not None:
                        found_id = str(found.get("element_id") or "")
                        if found_id and found_id != my_id:
                            sibling_elements[index] = found_id
                if not sibling_elements:
                    break
                row = _choice_row(my_id, elements_by_id, set(sibling_elements.values()))
                clause_id = element_entry.element_id
                clause_in_row = bool(clause_id) and (
                    clause_id == row or row in _ancestor_ids(clause_id, elements_by_id)
                )
                if not clause_in_row:
                    break
                current = _row_content(row, observation_elements, elements_by_id)
                if not current:
                    break
                row_parent = str((elements_by_id.get(row) or {}).get("parent_id") or "")
                if not row_parent:
                    break
                differing = []
                for index, sibling_id in sorted(sibling_elements.items()):
                    sibling_row = _choice_row(
                        sibling_id,
                        elements_by_id,
                        ({my_id} | set(sibling_elements.values())) - {sibling_id},
                    )
                    # Structural parallelism: a genuine alternative lives in the
                    # same list container as this action's row. A form button
                    # whose only same-type siblings are global navigation keys
                    # proves pane-vs-navbar, not a slot-determined choice.
                    sibling_parent = str(
                        (elements_by_id.get(sibling_row) or {}).get("parent_id") or ""
                    )
                    if sibling_parent != row_parent:
                        continue
                    sibling_content = _row_content(
                        sibling_row, observation_elements, elements_by_id
                    )
                    if sibling_content and sibling_content != current:
                        differing.append(index)
                if differing:
                    return RequirednessEvidence(
                        outcome=CounterfactualOutcome.TARGET_UNDERDETERMINED,
                        refs=_requiredness_refs(
                            evidence,
                            sibling_indexes=differing,
                            source_element=predicate.element,
                            effect_id="resolved_action_target",
                        ),
                    )
                break
    return RequirednessEvidence(outcome=CounterfactualOutcome.EVIDENCE_INSUFFICIENT)


def _classify_slot_requiredness(
    slot: IntentSlot,
    predicates: list[PredicateSpec],
    evidence: GuardEvidence,
) -> IntentSlot:
    evidence_items = [
        _classify_clause_requiredness(predicate, slot, evidence) for predicate in predicates
    ]
    outcomes = {item.outcome for item in evidence_items}
    if CounterfactualOutcome.PAYLOAD_UNDERDETERMINED in outcomes:
        selected = next(
            item
            for item in evidence_items
            if item.outcome is CounterfactualOutcome.PAYLOAD_UNDERDETERMINED
        )
        requiredness = SlotRequiredness.REQUIRED
    elif CounterfactualOutcome.TARGET_UNDERDETERMINED in outcomes:
        selected = next(
            item
            for item in evidence_items
            if item.outcome is CounterfactualOutcome.TARGET_UNDERDETERMINED
        )
        requiredness = SlotRequiredness.REQUIRED
    elif outcomes == {CounterfactualOutcome.ACTION_REMAINS_VALID}:
        selected = evidence_items[0]
        requiredness = SlotRequiredness.OPTIONAL
    else:
        # No machine rule proves the slot selects the target, payload, or
        # effect, so its absence cannot invalidate the action: optional. The
        # LLM's own requiredness claim is advisory metadata only and never
        # participates in this decision.
        selected = RequirednessEvidence(
            outcome=CounterfactualOutcome.ACTION_REMAINS_VALID,
            refs=_requiredness_refs(evidence),
        )
        requiredness = SlotRequiredness.OPTIONAL
    return slot.model_copy(
        update={
            "requiredness": requiredness,
            "requiredness_evidence": selected,
        },
        deep=True,
    )


def _canonical_action_target_binding_error(
    predicate: PredicateSpec,
    slot: IntentSlot,
    evidence: GuardEvidence,
    kind: GuardKind,
) -> str:
    """Reject target-identity bindings lacking deterministic choice evidence."""
    expected = predicate.expected
    if (
        kind is GuardKind.NAVIGATION
        and predicate.predicate_type == "action"
        and predicate.property in {"target_text", "target_content_desc"}
        and (predicate.operator or "==") in {"==", "contains"}
        and expected is not None
        and expected.kind == "intent"
    ):
        return (
            f"navigation intent slot '{slot.name}' restates fixed canonical action target identity"
        )
    if not _is_canonical_action_target_binding(predicate):
        return ""
    classification = _classify_clause_requiredness(predicate, slot, evidence)
    if classification.outcome is CounterfactualOutcome.TARGET_UNDERDETERMINED:
        return ""
    return (
        f"intent slot '{slot.name}' restates canonical action identity without "
        "machine-derived target underdetermination"
    )


def _action_target_surface_aliases(evidence: GuardEvidence) -> set[str]:
    """Return source aliases presented as the selected action target or its surface."""
    return {
        alias
        for alias in (
            evidence.action_target_alias,
            evidence.action_semantic_surface_alias,
        )
        if alias
    }


def _fixed_navigation_source_target_binding_error(
    predicate: PredicateSpec,
    slot: IntentSlot,
    evidence: GuardEvidence,
    kind: GuardKind,
) -> str:
    """Reject positive intent bindings over a fixed navigation target widget."""
    expected = predicate.expected
    if (
        kind is not GuardKind.NAVIGATION
        or expected is None
        or expected.kind != "intent"
        or (predicate.operator or "==") not in {"==", "contains"}
        or not predicate.element
    ):
        return ""
    if predicate.predicate_type == "read":
        if predicate.property not in _READABLE_STRING_PROPS:
            return ""
    elif predicate.predicate_type != "value":
        return ""

    element_alias = (
        predicate.element
        if predicate.element in evidence.source_registry.entries
        else evidence.source_registry.resource_id_to_alias.get(predicate.element)
    )
    if element_alias not in _action_target_surface_aliases(evidence):
        return ""
    return (
        f"navigation intent slot '{slot.name}' restates the fixed source action target "
        f"'{element_alias}'"
    )


def _enum_domain_error(slot: IntentSlot, evidence: GuardEvidence) -> str:
    """Require every closed-set value to be present in concrete source/action evidence."""
    observed = {
        value.strip()
        for entry in evidence.source_registry.entries.values()
        for value in (
            entry.text,
            entry.content_description,
            *entry.descendant_texts,
        )
        if value and value.strip()
    }
    observed.update(
        str(evidence.action.get(key) or "").strip()
        for key in ("target_text", "target_content_desc")
        if evidence.action.get(key)
    )
    unsupported = [value for value in slot.value_domain if value.strip() not in observed]
    if unsupported:
        return (
            f"enum slot '{slot.name}' has value_domain entries absent from source/action "
            f"evidence: {unsupported}"
        )
    return ""


def _source_observation_validation(
    predicate: PredicateSpec,
    compiled: str,
    slot: IntentSlot | None,
    evidence: GuardEvidence,
    resource_id: str | None,
) -> tuple[GuardAdmissionStatus | None, str]:
    """Replay one source predicate over every provenance observation.

    Intent values are unavailable offline, so intent clauses validate selector/property
    executability and numeric normalization. Literal clauses additionally must evaluate
    true on every observation. ``resource_id`` is the runtime-resolvable key produced by
    predicate lowering.
    """
    expected = predicate.expected
    if predicate.predicate_type == "action":
        if expected is not None and expected.kind == "literal":
            return (
                GuardAdmissionStatus.REJECTED,
                "literal action predicates restate canonical action identity",
            )
        return None, ""

    if not evidence.source_observations:
        return (
            GuardAdmissionStatus.REJECTED,
            "no provenance source observation is available for predicate replay",
        )
    if (
        expected is not None
        and expected.kind == "literal"
        and isinstance(expected.value, str)
        and expected.value in evidence.exploration_payload_values
    ):
        return (
            GuardAdmissionStatus.REJECTED,
            "literal is an input payload introduced by offline exploration provenance",
        )

    for observation in evidence.source_observations:
        screen_id = str(observation.get("screen_id") or "<unknown>")
        matches = [
            element
            for element in runtime_elements_from_observation(observation)
            if str(element.get("resource_id") or "") == resource_id
        ]
        if len(matches) != 1:
            return (
                GuardAdmissionStatus.REJECTED,
                f"resource_id {resource_id!r} resolves {len(matches)} times in "
                f"source observation {screen_id}",
            )
        context = screen_context_from_observation(observation)
        properties = context.elements[resource_id]

        if predicate.predicate_type == "read" and predicate.property not in properties:
            return (
                GuardAdmissionStatus.REJECTED,
                f"property {predicate.property!r} is unavailable in source observation {screen_id}",
            )
        if predicate.predicate_type == "count" and "children_count" not in properties:
            return (
                GuardAdmissionStatus.REJECTED,
                f"count is unavailable in source observation {screen_id}",
            )

        if slot is not None and slot.slot_type is SlotType.NUMBER:
            if predicate.predicate_type == "count":
                actual = properties["children_count"]
            elif predicate.predicate_type == "read":
                actual = properties[predicate.property or ""]
            else:
                actual = properties["value"]
            try:
                normalized = normalize_numeric_scalar(
                    actual,
                    source_value=True,
                )
            except NumericNormalizationError as exc:
                return (
                    GuardAdmissionStatus.REJECTED,
                    f"numeric source is not normalizable in observation {screen_id}: {exc}",
                )
            if predicate.predicate_type == "count" and (
                normalized < 0 or normalized != normalized.to_integral_value()
            ):
                return (
                    GuardAdmissionStatus.REJECTED,
                    f"count source is not a non-negative integer in observation {screen_id}",
                )

        if expected is None or expected.kind != "literal":
            continue
        result = _evaluator().evaluate(compiled, screen_ctx=context)
        if result.status is not GuardStatus.TRUE:
            return (
                GuardAdmissionStatus.REJECTED,
                f"literal predicate is {result.status.value} in source observation "
                f"{screen_id}: {result.failure_reason or 'condition not satisfied'}",
            )

    return None, ""


JOINT_UNSAT_REASON = "jointly_unsatisfiable_on_evidence"


@dataclass(frozen=True)
class _SlotBinding:
    """One admitted clause that pins an intent slot to a source-screen reading."""

    clause_index: int
    slot_name: str
    slot_type: SlotType
    compiled: str
    resource_id: str
    predicate: PredicateSpec


def _clause_source_reading(
    predicate: PredicateSpec,
    resource_id: str,
    observation: dict[str, object],
) -> tuple[object | None, bool]:
    """Read the source scalar this predicate compares against, if it is available.

    Returns ``(value, available)``. ``available`` is False when the element or property
    is missing from this observation, which makes the clause inconclusive here rather
    than false.
    """
    context = screen_context_from_observation(observation)
    properties = context.elements.get(resource_id)
    if properties is None:
        return None, False
    if predicate.predicate_type == "count":
        key = "children_count"
    elif predicate.predicate_type == "read":
        key = predicate.property or ""
    else:
        key = "value"
    if key not in properties:
        return None, False
    return properties[key], True


def _equality_key(value: object, slot_type: SlotType) -> tuple[object, bool]:
    """Normalize a reading the way the runtime evaluator compares it.

    Returns ``(key, comparable)``. Numbers go through the shared ``Decimal``
    normalization — no unit conversion, per the number-slot mandate — and text through
    the evaluator's own text normalization, so this check can never disagree with the
    comparison the shield will perform.
    """
    if slot_type is SlotType.NUMBER:
        try:
            return normalize_numeric_scalar(value, source_value=True), True
        except NumericNormalizationError:
            return None, False
    if isinstance(value, bool):
        return value, True
    if isinstance(value, str):
        return normalize_text_for_compare(value), True
    if value is None:
        return None, False
    return str(value), True


def _joint_satisfiability_error(
    bindings: list[_SlotBinding],
    evidence: GuardEvidence,
) -> str:
    """Reject a conjunction no single intent assignment can satisfy.

    Clause-level admission proves each predicate is *executable*; it never asks whether
    the clauses can hold *together*. A contract that equates one slot to two different
    source readings (the amount field recorded in cents and the preview recorded in
    dollars, say) admits clause by clause and then denies its transition forever,
    because no value the agent can bind satisfies both.

    The check is deliberately one-directional and evidence-bound: a contradiction must be
    *proven* on a recorded source observation where every participating clause is
    readable. If any participating clause is unreadable on an observation, that
    observation is inconclusive and ignored; if no observation is conclusive, the
    contract admits exactly as it did before. It can therefore only ever turn
    ``ADMITTED`` into ``REJECTED`` at proposal time — it never relaxes a guard, weakens
    an admitted clause, or changes runtime evaluation.

    Only ``==`` clauses pin a slot, so this proves nothing about inequality-only
    conjunctions (``> $intent.x`` with ``< $intent.x``); that is left inconclusive rather
    than approximated with interval reasoning.
    """
    by_slot: dict[str, list[_SlotBinding]] = {}
    for binding in bindings:
        by_slot.setdefault(binding.slot_name, []).append(binding)
    contested = {name: group for name, group in by_slot.items() if len(group) > 1}
    if not contested:
        return ""

    conclusive_observations = 0
    first_conflict = ""
    for observation in evidence.source_observations:
        screen_id = str(observation.get("screen_id") or "<unknown>")
        conflict = ""
        conclusive = True
        for slot_name, group in sorted(contested.items()):
            implied: dict[object, _SlotBinding] = {}
            for binding in group:
                value, available = _clause_source_reading(
                    binding.predicate, binding.resource_id, observation
                )
                if not available:
                    conclusive = False
                    break
                key, comparable = _equality_key(value, binding.slot_type)
                if not comparable:
                    conclusive = False
                    break
                if key in implied:
                    continue
                implied[key] = binding
            if not conclusive:
                break
            if len(implied) > 1 and not conflict:
                (value_a, binding_a), (value_b, binding_b) = sorted(
                    implied.items(), key=lambda item: item[1].clause_index
                )[:2]
                conflict = (
                    f"{JOINT_UNSAT_REASON}: intent slot '{slot_name}' must equal both "
                    f"{value_a!r} and {value_b!r} on source observation {screen_id} "
                    f"({binding_a.compiled} vs {binding_b.compiled})"
                )
        if not conclusive:
            continue
        conclusive_observations += 1
        if not conflict:
            # One recorded screen satisfies the whole conjunction: nothing is proven
            # contradictory, so the contract admits.
            return ""
        first_conflict = first_conflict or conflict
    if conclusive_observations and first_conflict:
        return first_conflict
    return ""


def admit_guard_contract(
    contract: GuardContract,
    evidence: GuardEvidence,
) -> GuardAdmissionResult:
    """Admit a contract clause-by-clause, then validate contract-level coherence."""
    if contract.kind is GuardKind.UNKNOWN:
        return GuardAdmissionResult(
            admitted=False,
            status=GuardAdmissionStatus.REJECTED,
            reason="transition semantic kind is unresolved",
        )

    # Every transition is analyzed by the semantic generator. An empty conjunction is
    # valid only when that analysis cites the complete machine-resolvable classification
    # surface. Navigation is a semantic classification, not an action-type shortcut:
    # navigation contracts may also carry ordinary clauses and are processed below.
    if not contract.slots and not contract.clauses:
        if contract.kind in {GuardKind.NONE, GuardKind.NAVIGATION}:
            evidence_error = _semantic_evidence_error(
                contract,
                evidence,
                require_complete=True,
            )
            if evidence_error:
                return GuardAdmissionResult(
                    admitted=False,
                    status=GuardAdmissionStatus.REJECTED,
                    reason=evidence_error,
                )
            return GuardAdmissionResult(
                admitted=True,
                status=GuardAdmissionStatus.ADMITTED,
                reason="evidence-backed semantic analysis found no transition-specific clause",
            )
        return GuardAdmissionResult(
            admitted=False,
            status=GuardAdmissionStatus.REJECTED,
            reason="semantic guard incomplete: no executable clause",
        )

    semantic_evidence_error = _semantic_evidence_error(
        contract,
        evidence,
        require_complete=False,
    )
    if semantic_evidence_error:
        return GuardAdmissionResult(
            admitted=False,
            status=GuardAdmissionStatus.REJECTED,
            reason=semantic_evidence_error,
        )

    registry = evidence.source_registry
    declared: dict[str, IntentSlot] = {}
    duplicate_slots: list[str] = []
    for slot in contract.slots:
        if slot.name in declared:
            duplicate_slots.append(slot.name)
        declared[slot.name] = slot
    if duplicate_slots:
        names = ", ".join(sorted(set(duplicate_slots)))
        return GuardAdmissionResult(
            admitted=False,
            status=GuardAdmissionStatus.REJECTED,
            reason=f"duplicate intent slot declaration(s): {names}",
        )

    slot_errors: list[str] = []
    for slot in contract.slots:
        if not slot.name.strip():
            slot_errors.append("intent slot name must be non-empty")
        if slot.slot_type is SlotType.ENUM and not slot.value_domain:
            slot_errors.append(f"enum slot '{slot.name}' requires a non-empty value_domain")
        if slot.slot_type is SlotType.ENUM and slot.value_domain:
            stripped_domain = [value.strip() for value in slot.value_domain]
            if any(not value for value in stripped_domain):
                slot_errors.append(
                    f"enum slot '{slot.name}' value_domain entries must be non-empty"
                )
            if len(set(stripped_domain)) != len(stripped_domain):
                slot_errors.append(f"enum slot '{slot.name}' value_domain entries must be unique")
            domain_error = _enum_domain_error(slot, evidence)
            if domain_error:
                slot_errors.append(domain_error)
    if slot_errors:
        return GuardAdmissionResult(
            admitted=False,
            status=GuardAdmissionStatus.REJECTED,
            reason="invalid intent slot schema: " + "; ".join(slot_errors),
            rejected_predicates=slot_errors,
        )

    classification_errors: list[str] = []
    referenced_slots: set[str] = set()
    clause_slots: list[str | None] = []
    for clause in contract.clauses:
        expected = clause.predicate.expected
        predicate_slot = (
            expected.slot
            if expected is not None and expected.kind == "intent" and expected.slot
            else None
        )
        if clause.slot is not None and clause.slot != predicate_slot:
            classification_errors.append(
                f"clause slot {clause.slot!r} conflicts with predicate slot {predicate_slot!r}"
            )
        slot_name = predicate_slot or clause.slot
        clause_slots.append(slot_name)
        if slot_name is None:
            continue
        referenced_slots.add(slot_name)
    undeclared = sorted(referenced_slots - set(declared))
    unreferenced = sorted(set(declared) - referenced_slots)
    if undeclared:
        classification_errors.append(
            "clause references undeclared intent slot(s): " + ", ".join(undeclared)
        )
    if unreferenced:
        classification_errors.append(
            "declared intent slot(s) have no guard clause: " + ", ".join(unreferenced)
        )
    if classification_errors:
        return GuardAdmissionResult(
            admitted=False,
            status=GuardAdmissionStatus.REJECTED,
            reason="invalid slot/clause linkage: " + "; ".join(classification_errors),
            rejected_predicates=classification_errors,
        )

    predicates_by_slot: dict[str, list[PredicateSpec]] = {name: [] for name in declared}
    for clause, slot_name in zip(contract.clauses, clause_slots, strict=True):
        if slot_name is not None:
            predicates_by_slot[slot_name].append(clause.predicate)
    classified_slots = [
        _classify_slot_requiredness(
            slot,
            predicates_by_slot[slot.name],
            evidence,
        )
        for slot in contract.slots
    ]
    declared = {slot.name: slot for slot in classified_slots}

    surviving: list[str] = []
    rejected: list[str] = []
    processed_clauses: list[GuardClause] = []
    slot_bindings: list[_SlotBinding] = []
    for clause_index, (clause, slot_name) in enumerate(
        zip(contract.clauses, clause_slots, strict=True)
    ):
        admitted_clause = clause.model_copy(deep=True)
        admitted_clause.slot = slot_name
        slot_spec = declared[slot_name] if slot_name is not None else None

        pred = admitted_clause.predicate
        result = _lower_predicate(pred, registry, declared)
        if result.compiled is None:
            admitted_clause.admission_status = GuardAdmissionStatus.REJECTED
            admitted_clause.admission_reason = result.reason
            processed_clauses.append(admitted_clause)
            rejected.append(result.reason)
            continue
        # Literal clauses are parsed by their mandatory source-observation evaluation.
        # Intent clauses cannot be evaluated offline, so check their syntax here.
        if pred.expected is not None and pred.expected.kind == "intent":
            try:
                _parser().parse(result.compiled)
            except LarkError as exc:
                reason = f"parse error: {exc}"
                admitted_clause.admission_status = GuardAdmissionStatus.REJECTED
                admitted_clause.admission_reason = reason
                processed_clauses.append(admitted_clause)
                rejected.append(reason)
                continue

        admitted_clause.compiled_expr = result.compiled
        observation_status, observation_error = _source_observation_validation(
            pred,
            result.compiled,
            slot_spec,
            evidence,
            result.resource_id,
        )
        if observation_status is not None:
            admitted_clause.admission_status = observation_status
            admitted_clause.admission_reason = observation_error
            processed_clauses.append(admitted_clause)
            rejected.append(observation_error)
            continue
        identity_error = (
            _canonical_action_target_binding_error(
                pred,
                slot_spec,
                evidence,
                contract.kind,
            )
            if slot_spec is not None
            else ""
        )
        if not identity_error and slot_spec is not None:
            identity_error = _fixed_navigation_source_target_binding_error(
                pred,
                slot_spec,
                evidence,
                contract.kind,
            )
        if identity_error:
            admitted_clause.admission_status = GuardAdmissionStatus.REJECTED
            admitted_clause.admission_reason = identity_error
            rejected.append(identity_error)
        else:
            admitted_clause.admission_status = GuardAdmissionStatus.ADMITTED
            admitted_clause.admission_reason = "admitted"
            surviving.append(result.compiled)
            # Record the clauses that pin a slot to one source reading, for the
            # contract-level joint-satisfiability check below.
            if (
                slot_spec is not None
                and slot_name is not None
                and result.resource_id
                and pred.operator == "=="
                and pred.predicate_type in ("read", "value", "count")
            ):
                slot_bindings.append(
                    _SlotBinding(
                        clause_index=clause_index,
                        slot_name=slot_name,
                        slot_type=slot_spec.slot_type,
                        compiled=result.compiled,
                        resource_id=result.resource_id,
                        predicate=pred,
                    )
                )
        processed_clauses.append(admitted_clause)

    # Any rejected predicate is genuinely non-executable -> hard reject (do not partially
    # attach). This covers unresolved aliases, unsupported action properties, undeclared
    # intent slots, non-readable properties, and literals proven false from source
    # evidence.
    if rejected:
        return GuardAdmissionResult(
            admitted=False,
            status=GuardAdmissionStatus.REJECTED,
            guard=None,
            reason="rejected non-executable predicate(s): " + "; ".join(rejected),
            rejected_predicates=rejected,
            clauses=processed_clauses,
            slots=classified_slots,
        )

    # Every clause is individually executable; the conjunction still has to be
    # satisfiable by one intent assignment. A contract that cannot be is not a strict
    # guard, it is a permanent DENY on a modeled transition.
    joint_error = _joint_satisfiability_error(slot_bindings, evidence)
    if joint_error:
        conflicting = {
            binding.clause_index
            for binding in slot_bindings
            if f"'{binding.slot_name}'" in joint_error
        }
        for index in conflicting:
            processed_clauses[index].admission_status = GuardAdmissionStatus.REJECTED
            processed_clauses[index].admission_reason = joint_error
        return GuardAdmissionResult(
            admitted=False,
            status=GuardAdmissionStatus.REJECTED,
            guard=None,
            reason=joint_error,
            rejected_predicates=[joint_error],
            clauses=processed_clauses,
            slots=classified_slots,
        )

    guard = " && ".join(surviving)

    return GuardAdmissionResult(
        admitted=True,
        status=GuardAdmissionStatus.ADMITTED,
        guard=guard,
        reason=f"admitted: {len(surviving)} executable predicate(s)",
        clauses=processed_clauses,
        slots=classified_slots,
    )

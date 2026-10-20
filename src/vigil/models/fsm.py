"""Per-app extended finite-state machine ``M_A`` and its supporting evidence.

This is the central data structure of Vigil. It wraps a networkx DiGraph and provides
methods for state/transition management, topological verification, and serialization.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Hashable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

from vigil.models.evolution import EvolutionOperation, EvolutionOutcome
from vigil.models.guard import GuardContract

# State/transition provenance markers for conservative runtime evolution.
# ``construction_source`` on a state and ``evidence_source`` on a transition's
# origin provenance entry are plain strings; these constants name the two runtime
# values so the verifier and evolution engine agree without a new schema field.
#
# A state built from an unknown runtime observation carries
# ``evidence.construction_source == RUNTIME_EVOLUTION_SOURCE``; the verifier reads
# this as "coverage-incomplete" so a missing outgoing edge is refused
# rather than DENY. A transition first created from a runtime fallback trace carries
# ``FALLBACK_PROVENANCE_SOURCE`` as its *origin* provenance ``evidence_source``.
RUNTIME_EVOLUTION_SOURCE = "runtime_evolution"
FALLBACK_PROVENANCE_SOURCE = "runtime_fallback"

# FSM JSON schema versions. v5 drops the scalar replay-confidence axis: transition
# support is determined by membership in the accepted transition graph.
# ``deserialize`` still reads v4 by migrating the payload in memory via
# :func:`migrate_v4_payload`.
# v6 is additive: ``StateIdentity.member_fingerprints`` records every live-screen
# identity a quotiented state covers. A v5 payload reads as v6 with that list empty,
# in which case localization behaves exactly as it did before.
SCHEMA_VERSION = "6"
SUPPORTED_SCHEMA_VERSIONS = ("4", "5", "6")

# Keys dropped when a v4 payload is migrated. The transition list holds the removed
# confidence axis plus guard-era extras that predate ``guard_contract``; both are
# rejected by ``extra="forbid"`` and only ever existed in stored artifacts.
_REMOVED_TRANSITION_KEYS = (
    "confidence",
    "replay_success_count",
    "replay_trial_count",
    "requires_guard",
    "guard_admission_status",
    "guard_admission_reason",
)
_REMOVED_STATE_KEYS = ("kind", "legacy_invariants")
_REMOVED_PAYLOAD_KEYS = ("version", "semantic_revision")
# Provenance sources that no longer have a producer anywhere in the pipeline.
_DEAD_EVIDENCE_SOURCES = frozenset({"inferred_dialog", "inferred_tab"})


class HierarchyLevel(StrEnum):
    """Activity, fragment, and component scopes for state abstraction."""

    ACTIVITY = "activity"
    FRAGMENT = "fragment"
    COMPONENT = "component"


class ContainerType(StrEnum):
    """Classification for scrollable containers within a state.

    DYNAMIC: Structure fixed but content/count varies across visits (e.g.,
        WiFi list — different networks at different locations). Determined by
        range/pattern invariants from multi-visit observation.
    NONE: State does not contain a classified scrollable container.
    """

    DYNAMIC = "dynamic"
    NONE = "none"


class TransitionLookupStatus(StrEnum):
    """Result of resolving an action against a state's outgoing transitions."""

    MATCH = "match"
    NO_MATCH = "no_match"
    UNCERTAIN = "uncertain"


_GLOBAL_ACTION_TYPES = frozenset({"navigate_back", "navigate_home"})
_VALUE_PAYLOAD_ACTION_TYPES = frozenset({"input_text"})
_VALUE_PAYLOAD_FIELDS = frozenset({"text", "value", "input_text"})
_ACTION_KEY_FIELDS = (
    "type",
    "resource_id",
    "target_resource_id",
    "target_text",
    "target_content_desc",
    "target_class",
    "target_class_name",
    "target_selector",
)
_ACTION_IDENTITY_FIELDS = tuple(
    k for k in (*_ACTION_KEY_FIELDS, "target", *_VALUE_PAYLOAD_FIELDS) if k != "type"
)
_STABLE_TARGET_IDENTITY_FIELDS = frozenset(
    {
        "resource_id",
        "target_resource_id",
        "target_text",
        "target_content_desc",
    }
)
_TEMPLATE_ITEM_IDENTITY_FIELDS = frozenset(
    {
        "target",
        "target_text",
        "target_content_desc",
        "target_selector",
    }
)
_SELECTOR_IDENTITY_FIELDS = (
    "resource_id",
    "text",
    "content_description",
    "nearby_text",
    "class_name",
    "ancestor_chain",
)


def _normalize_action_value(value: Any) -> Hashable | None:
    """Normalize action identity values; empty strings/containers mean absent."""
    if value is None:
        return None
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, list | tuple):
        normalized = tuple(_normalize_action_value(v) for v in value)
        return normalized if any(v is not None for v in normalized) else None
    if isinstance(value, dict):
        normalized_items = tuple(
            sorted((str(k), _normalize_action_value(v)) for k, v in value.items())
        )
        return normalized_items if any(v is not None for _, v in normalized_items) else None
    if isinstance(value, bool | int | float):
        return value
    return str(value)


def _selector_signature(selector: Any) -> Hashable | None:
    """Stable selector identity; debug-only bounds/depth are intentionally excluded."""
    if not isinstance(selector, dict) or not selector:
        return None
    parts = tuple(
        (field, _normalize_action_value(selector.get(field))) for field in _SELECTOR_IDENTITY_FIELDS
    )
    return parts if any(value is not None for _, value in parts) else None


def _first_present(*values: Any) -> Hashable | None:
    for value in values:
        normalized = _normalize_action_value(value)
        if normalized is not None:
            return normalized
    return None


def _canonical_action_mapping(action: dict[str, Any]) -> dict[str, Hashable | None]:
    """Normalize every stable serialized action identity field into one mapping."""
    selector = action.get("target_selector") or {}
    selector_map = selector if isinstance(selector, dict) else {}

    resource_id = _first_present(
        action.get("resource_id"),
        action.get("target_resource_id"),
        selector_map.get("resource_id"),
    )
    target_resource_id = _first_present(
        action.get("target_resource_id"),
        action.get("resource_id"),
        selector_map.get("resource_id"),
    )
    target_class = _first_present(
        action.get("target_class"),
        action.get("target_class_name"),
        action.get("class_name"),
        selector_map.get("class_name"),
    )
    target_class_name = _first_present(
        action.get("target_class_name"),
        action.get("target_class"),
        action.get("class_name"),
        selector_map.get("class_name"),
    )
    text = _first_present(action.get("text"), action.get("value"), action.get("input_text"))
    value = _first_present(action.get("value"), action.get("text"), action.get("input_text"))

    return {
        "type": _first_present(action.get("type"), action.get("action_type")),
        "target": _first_present(action.get("target")),
        "resource_id": resource_id,
        "target_resource_id": target_resource_id,
        "target_text": _first_present(
            action.get("target_text"),
            selector_map.get("text"),
            selector_map.get("nearby_text"),
        ),
        "target_content_desc": _first_present(
            action.get("target_content_desc"),
            selector_map.get("content_description"),
        ),
        "target_class": target_class,
        "target_class_name": target_class_name,
        "target_selector": _selector_signature(selector),
        "text": text,
        "value": value,
    }


def canonical_transition_action(action: dict[str, Any]) -> dict[str, Any]:
    """Derive canonical action ``sigma_t`` from a source-bound raw action ``a_t``.

    Capture-local element handles, bounds, selector coordinates/depth, and
    exploration input samples are evidence. They belong in transition
    provenance, not in the EFSM action label. The stable selector identifies the
    target component; guardrails check concrete values for parameterized input actions.
    """
    mapping = _canonical_action_mapping(action)
    action_type = mapping.get("type")
    canonical: dict[str, Any] = {}
    if action_type is not None:
        canonical["type"] = action_type

    stable_target_fields = _identity_fields(mapping) & _STABLE_TARGET_IDENTITY_FIELDS
    for field in (
        "resource_id",
        "target_resource_id",
        "target_text",
        "target_content_desc",
        "target_class",
        "target_class_name",
    ):
        if action_type in _VALUE_PAYLOAD_ACTION_TYPES and field == "target_text":
            continue
        value = mapping.get(field)
        if value is not None:
            canonical[field] = value

    selector = action.get("target_selector")
    if not stable_target_fields and isinstance(selector, dict):
        stable_selector = {
            field: selector[field]
            for field in _SELECTOR_IDENTITY_FIELDS
            if _normalize_action_value(selector.get(field)) is not None
        }
        if stable_selector:
            canonical["target_selector"] = stable_selector

    if action_type not in _VALUE_PAYLOAD_ACTION_TYPES:
        for field in _VALUE_PAYLOAD_FIELDS:
            value = mapping.get(field)
            if value is not None:
                canonical[field] = value

    return canonical


def is_value_payload_action(action: dict[str, Any]) -> bool:
    """True for actions whose concrete payload is not stable transition identity."""
    return _canonical_action_mapping(action).get("type") in _VALUE_PAYLOAD_ACTION_TYPES


def canonical_action_key(action: dict[str, Any]) -> tuple[tuple[str, Hashable | None], ...]:
    """Return the stable identity of a canonical action in the alphabet ``Sigma``."""
    mapping = _canonical_action_mapping(canonical_transition_action(action))
    return tuple((field, mapping[field]) for field in (*_ACTION_KEY_FIELDS, "text", "value"))


def _identity_fields(action_key: dict[str, Hashable | None]) -> set[str]:
    return {field for field in _ACTION_IDENTITY_FIELDS if action_key.get(field) is not None}


def _transition_identity_fields(
    action_type: Hashable | None,
    action_key: dict[str, Hashable | None],
) -> set[str]:
    fields = _identity_fields(action_key)
    if action_type in _VALUE_PAYLOAD_ACTION_TYPES:
        return fields - _VALUE_PAYLOAD_FIELDS
    return fields


class StateIdentity(BaseModel):
    """Deterministic identity for an abstract state: functional vs. structural hashes.

    Several hashes coexist; ``StateLocator`` consults them in a fixed priority order
    when mapping a live screen to a stored state. Which hash wins, per scenario:

    1. **Refined states** resolve only by the atomic ``(refinement_family,
       refinement_signature)`` pair. A component/structural hash may *nominate* a
       family, but a near-miss or an ambiguous family is ``UNKNOWN`` — never a partial
       component match.
    2. **Ordinary states** resolve by ``structural_hash`` (a 16-char builder hash, also
       indexed by its 12-char live prefix), ``functional_hash`` (which may hold a hybrid
       id), or any entry in ``member_fingerprints`` (the other live-screen identities the
       state absorbed while quotienting); an exact hit is ``EXACT``.
    3. **Refined siblings** resolve by ``(base_fp, secondary_hash)``; an exact hit is
       ``EXACT``.
    4. A unique ``canonical_equivalence_hash`` hit is ``CANONICAL_EQUIVALENT`` and is
       verifier-usable.
    5. A ``functional_hash`` scan match is ``SIMILAR`` — non-authoritative: the verifier
       refuses on it and only evolution consumes it.

    When a refined family and an ordinary state disagree, the side whose stored
    ``canonical_equivalence_hash`` equals the live canonical hash wins; otherwise the
    result is ``UNKNOWN``. Keep this matrix in sync with ``symbolic/state_locator.py``.
    """

    functional_hash: str
    structural_hash: str | None = None
    canonical_equivalence_hash: str | None = None
    # Every live-screen identity this state was observed to cover (schema-v6 additive;
    # empty on v5 payloads, which then localize exactly as before). Quotienting absorbs
    # many observed identities into one state but keeps only the representative in
    # ``functional_hash``; without this list ``StateLocator`` can exact-match that one
    # representative and every other member screen falls through to the (frequently
    # ambiguous) canonical bucket. Sorted and deduplicated; derived from
    # ``evidence.raw_screen_ids`` by ``vigil.neuro.fsm_builder.assign_member_fingerprints``.
    member_fingerprints: list[str] = Field(default_factory=list)
    # Runtime Abstraction-refinement identity (schema-v4 additive; None for non-refined
    # states). ``refinement_signature`` is the deterministic schema/action-surface identity
    # and is unique only *within* one ``refinement_family`` (the split-source family), so
    # localization must resolve the pair atomically.
    refinement_signature: str | None = None
    refinement_family: str | None = None


class AndroidStateContext(BaseModel):
    """Android-specific observation context, kept separate from abstract identity."""

    activity_name: str | None = None
    package_name: str | None = None
    window_type: str | None = None


class StateEvidence(BaseModel):
    """Trace-derived evidence supporting that this state exists.

    XML/runtime traces are the deterministic source of truth — this is where
    the supporting screen ids live. ``observation_count`` is a derived
    field over ``raw_screen_ids`` so mutating the underlying list (e.g.
    appending newly-observed screens during builder merging) can never
    leave the count stale.
    """

    raw_screen_ids: list[str] = Field(default_factory=list)
    construction_source: str = "observed_trace"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def observation_count(self) -> int:
        """Return the number of observations supporting this abstract state."""
        return len(self.raw_screen_ids)


class StateAbstraction(BaseModel):
    """Dynamic container / template metadata for this state."""

    container_type: ContainerType = ContainerType.NONE
    container_selector: dict[str, Any] = Field(default_factory=dict)
    template_id: str | None = None
    parameter_schema: dict[str, str] = Field(default_factory=dict)


class StateInvariant(BaseModel):
    """One admitted clause of the state invariant ``I(s_t)`` with provenance."""

    model_config = ConfigDict(extra="forbid")

    expr: str
    source: str = "unknown"
    evidence_count: int = Field(default=0, ge=0)


class StateAnnotations(BaseModel):
    """LLM-derived, non-authoritative annotations.

    Per project rules these never decide state equality, edges, edge trust,
    or runtime verdicts. ``display_name`` is annotation-only —
    consumers must use ``AbstractState.name`` as the canonical state name;
    ``display_name`` does not override identity or routing.
    """

    display_name: str = ""
    alt_text: str = ""
    page_function: str = ""


class AbstractState(BaseModel):
    """An abstract GUI state ``s_t`` in the per-app state set ``S``.

    The schema is partitioned into nested canonical submodels:

    - ``identity`` (StateIdentity): functional / structural hashes.
    - ``android_context`` (AndroidStateContext): activity / package / window.
    - ``evidence`` (StateEvidence): trace-derived raw screen ids.
    - ``abstraction`` (StateAbstraction): container / template metadata.
    - ``invariant_specs`` (list[StateInvariant]): runtime-checkable invariants.
    - ``annotations`` (StateAnnotations): LLM-derived non-authoritative labels.

    Identity, evidence, annotations, and abstraction metadata use nested models.
    ``invariant_specs`` contains the admitted clauses of ``I(s_t)``.

    ``name`` is the canonical state name. ``annotations.display_name`` is
    annotation-only and must not override identity or routing.
    """

    model_config = ConfigDict(extra="forbid")

    state_id: str
    name: str
    hierarchy_level: HierarchyLevel
    parent_state: str | None = None
    identity: StateIdentity
    android_context: AndroidStateContext = Field(default_factory=AndroidStateContext)
    evidence: StateEvidence = Field(default_factory=StateEvidence)
    abstraction: StateAbstraction = Field(default_factory=StateAbstraction)
    invariant_specs: list[StateInvariant] = Field(default_factory=list)
    annotations: StateAnnotations = Field(default_factory=StateAnnotations)


class ProvenanceEntry(BaseModel):
    """Single evidence record explaining how a transition entered the FSM.

    A transition may carry multiple entries when independent observations support
    the same canonical action and successor state.

    Attributes:
        trace_step_index: Position in the source trace ``traces`` array; ``-1``
            for entries without a trace position.
        source_screen_id: Raw screen id captured before the action.
        target_screen_id: Raw screen id captured after the action.
        raw_action: Trace/evidence action sample before verifier-level canonicalization.
        evidence_source: Where the supporting evidence came from. One of
            ``"observed" | "identity_inconsistent" | "runtime_fallback"``.
            ``"runtime_fallback"`` on the *first* provenance entry marks a transition
            originally created by conservative runtime evolution (see
            ``is_runtime_fallback_origin``).
    """

    trace_step_index: int = -1
    source_screen_id: str | None = None
    target_screen_id: str | None = None
    raw_action: dict[str, Any] = Field(default_factory=dict)
    evidence_source: str = "observed"


class Transition(BaseModel):
    """An observed abstract transition ``(s_t, sigma_t, s_{t+1})`` in the EFSM.

    Attributes:
        source: Source state ``s_t`` identifier.
        target: Successor state ``s_{t+1}`` identifier.
        action: Canonical GUI action ``sigma_t`` for this edge. Capture-local
            handles, bounds, and typed input samples live in provenance raw_action.
        guard: Derived aggregate DSL expression for audit/export only.
        ambiguous_evidence: Whether this edge's evidence fails to attribute it to
            exactly this ``(source, canonical action)`` pair -- an observation scope
            that only partly covers the app, an action sample that resolved
            inconsistently, a determinism conflict between indistinguishable source
            screens, or evidence that could not be pinned to one sibling after a
            split. Consumed ONLY by offline construction: state refinement and the
            determinism guard must not draw conclusions from it. It is **not** a
            trust modifier and has no effect on any runtime verdict. The accepted
            transition itself is the model's statement of support.
        observed_count: Number of times this transition was observed during exploration.
        provenance: Evidence records explaining where this transition came from.
        guard_contract: Typed transition guardrail ``Gamma(s_t, sigma_t)``. Runtime
            verification activates independently admitted clauses from this contract.
            ``guard`` is only a derived aggregate for audit/export.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    action: dict[str, Any]
    guard: str | None = None
    ambiguous_evidence: bool = False
    observed_count: int = 0
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
    guard_contract: GuardContract | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_low_trust(cls, data: Any) -> Any:
        """Accept ``low_trust`` as the former spelling of ``ambiguous_evidence``.

        Model-level migration covers direct construction as well as bundle loading.
        """
        if isinstance(data, dict) and "low_trust" in data:
            data = {**data, "ambiguous_evidence": data["low_trust"]}
            data.pop("low_trust")
        return data


def is_runtime_fallback_origin(transition: Transition) -> bool:
    """True only when a transition was *originally created* by runtime evolution.

    The origin is the first provenance entry. An existing trusted edge may later
    receive runtime-fallback corroboration in a subsequent provenance entry; that
    must not downgrade the edge, so this checks the origin only — never ``any(...)``.
    Used for evolution provenance; it never changes an execution-time verdict.
    """
    if not transition.provenance:
        return False
    return transition.provenance[0].evidence_source == FALLBACK_PROVENANCE_SOURCE


def _payload_values(container: Any) -> list[Any]:
    """Values of a payload collection stored either as a dict or a list."""
    if isinstance(container, dict):
        return list(container.values())
    if isinstance(container, list):
        return container
    return []


def _migrate_transition_payload(transition: dict[str, Any]) -> None:
    for removed in _REMOVED_TRANSITION_KEYS:
        transition.pop(removed, None)
    for entry in _payload_values(transition.get("provenance")):
        if not isinstance(entry, dict):
            continue
        legacy_source = entry.pop("confidence_source", None)
        source = entry.get("evidence_source", legacy_source)
        if source is None:
            continue
        entry["evidence_source"] = "observed" if source in _DEAD_EVIDENCE_SOURCES else source


def _migrate_state_payload(state: dict[str, Any]) -> None:
    for removed in _REMOVED_STATE_KEYS:
        state.pop(removed, None)
    for invariant in _payload_values(state.get("invariant_specs")):
        if isinstance(invariant, dict):
            invariant.pop("confidence", None)


def migrate_v4_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate a schema-v4 FSM payload to the v5 shape, in place-safe fashion.

    v5 removes the scalar replay-confidence axis, renames the provenance
    ``confidence_source`` key to ``evidence_source``, and folds the two dead
    ``inferred_*`` sources back into ``"observed"``. It also strips model keys that
    were removed earlier (state ``kind``/``legacy_invariants``, invariant
    ``confidence``, guard-era transition extras) and the unread top-level ``version``
    / superseded ``semantic_revision`` fields, so stale artifacts load again under
    ``extra="forbid"``.

    Used by :meth:`AppFSM.deserialize`. It mutates and returns ``data`` and is
    idempotent for payloads already migrated to the current schema.
    """
    data["schema_version"] = SCHEMA_VERSION
    for removed in _REMOVED_PAYLOAD_KEYS:
        data.pop(removed, None)

    transition_groups: list[Any] = [data.get("transitions")]
    state_groups: list[Any] = [data.get("states")]
    for template in _payload_values(data.get("sub_fsm_templates")):
        if isinstance(template, dict):
            state_groups.append(template.get("states"))
            transition_groups.append(template.get("transitions"))

    for group in transition_groups:
        for transition in _payload_values(group):
            if isinstance(transition, dict):
                _migrate_transition_payload(transition)
    for group in state_groups:
        for state in _payload_values(group):
            if isinstance(state, dict):
                _migrate_state_payload(state)

    return data


def transition_identity_key(
    transition: Transition,
    *,
    source: str | None = None,
    target: str | None = None,
) -> tuple[Any, ...]:
    """Identity for one accepted source/action/successor transition."""
    return (
        source if source is not None else transition.source,
        target if target is not None else transition.target,
        canonical_action_key(transition.action),
    )


def record_efsm_revision(
    fsm: AppFSM,
    operation: EvolutionOperation,
    event: str,
    **fields: Any,
) -> int:
    """The single commit path for an effective, runtime-verification-affecting change.

    Increments ``efsm_revision`` exactly once and records the model operation.
    ``event`` supplies lower-level audit detail. Returns the new revision.
    """
    revision_before = fsm.efsm_revision
    fsm.efsm_revision = revision_before + 1
    entry: dict[str, Any] = {
        "operation": operation.value,
        "outcome": EvolutionOutcome.APPLIED.value,
        "event": event,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "efsm_revision_before": revision_before,
        "efsm_revision_after": fsm.efsm_revision,
    }
    entry.update(fields)
    fsm.evolution_log.append(entry)
    return fsm.efsm_revision


def record_evolution_audit(
    fsm: AppFSM,
    operation: EvolutionOperation,
    outcome: EvolutionOutcome,
    event: str,
    **fields: Any,
) -> None:
    """Record a non-revision-bearing evolution attempt or evidence update."""
    entry: dict[str, Any] = {
        "operation": operation.value,
        "outcome": outcome.value,
        "event": event,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "efsm_revision_before": fsm.efsm_revision,
        "efsm_revision_after": fsm.efsm_revision,
    }
    entry.update(fields)
    fsm.evolution_log.append(entry)


@dataclass(frozen=True)
class TransitionLookup:
    """Resolved transition plus ambiguity status for action identity matching."""

    status: TransitionLookupStatus
    transition: Transition | None = None
    target_state_id: str | None = None
    details: str = ""


class SubFsmTemplate(BaseModel):
    """Parameterized sub-FSM for dynamic container item detail pages.

    When a DYNAMIC container's items all lead to structurally identical
    detail pages (verified via smart stopping in Stage 1), this template
    represents all N possible detail pages with a single parameterized state.
    """

    template_id: str
    source_state_id: str
    states: dict[str, AbstractState] = Field(default_factory=dict)
    transitions: list[Transition] = Field(default_factory=list)


class AppFSM:
    """Per-app EFSM ``M_A`` with transition guardrails and state invariants.

    Wraps a networkx.DiGraph where nodes are AbstractStates and edges are Transitions.
    ``states`` is ``S`` and ``initial_state`` is ``s_0``. Canonical labels on
    ``transitions`` form ``Sigma``; each transition's ``guard_contract`` supplies
    ``Gamma(s_t, sigma_t)`` and each state's ``invariant_specs`` supplies ``I(s_t)``.
    The transition list preserves distinct actions sharing the same endpoints.

    Args:
        app_package: Android package name (e.g., "com.android.settings").
    """

    def __init__(self, app_package: str) -> None:
        self.app_package = app_package
        self.graph: nx.DiGraph = nx.DiGraph()
        self.states: dict[str, AbstractState] = {}
        self.transitions: list[Transition] = []
        self.initial_state: str | None = None
        self.efsm_revision: int = 0
        self.construction_log: list[dict[str, Any]] = []
        self.evolution_log: list[dict[str, Any]] = []
        self.sub_fsm_templates: dict[str, SubFsmTemplate] = {}

    def add_state(self, state: AbstractState) -> None:
        """Add an abstract state to the FSM."""
        self.states[state.state_id] = state
        self.graph.add_node(state.state_id)

    def add_transition(self, transition: Transition) -> None:
        """Add a transition between two states."""
        transition.action = canonical_transition_action(transition.action)
        self.transitions.append(transition)
        self.graph.add_edge(transition.source, transition.target)

    def rebuild_graph(self) -> None:
        """Rebuild ``graph`` from the current ``states`` and ``transitions``.

        The labeled transition relation lives in ``transitions`` (a list — multiple
        actions may share endpoints, which a ``DiGraph`` would collapse). After an
        Abstraction-refinement state split rewrites ``states``/``transitions`` in one
        atomic step, this re-derives the reachability graph so the three stay
        consistent.
        """
        self.graph = nx.DiGraph()
        for state in self.states.values():
            self.graph.add_node(state.state_id)
        for transition in self.transitions:
            self.graph.add_edge(transition.source, transition.target)

    @staticmethod
    def _compatible_with_proposed_identity(
        stored_key: dict[str, Hashable | None],
        proposed_key: dict[str, Hashable | None],
        proposed_fields: set[str],
    ) -> bool:
        """True when all identity fields supplied by the proposal agree."""
        fields = proposed_fields
        stable_fields = fields & _STABLE_TARGET_IDENTITY_FIELDS
        if "target" in fields and stable_fields:
            # ``target`` is the capture-local element_id. Ignore churn in that
            # handle only when stable text/resource/content-desc identity also
            # binds the proposal to the stored action.
            fields = fields - {"target"}
        return all(stored_key.get(field) == proposed_key.get(field) for field in fields)

    @staticmethod
    def _item_specific_identity_fields(proposed_key: dict[str, Hashable | None]) -> set[str]:
        """Identity fields strong enough to bind a dynamic-template item."""
        return {
            field for field in _TEMPLATE_ITEM_IDENTITY_FIELDS if proposed_key.get(field) is not None
        }

    @staticmethod
    def _template_binding_missing_lookup() -> TransitionLookup:
        return TransitionLookup(
            status=TransitionLookupStatus.UNCERTAIN,
            details="template_binding_missing",
        )

    def _template_conflict_lookup(
        self,
        candidates: list[Transition],
        proposed_key: dict[str, Hashable | None],
        item_fields: set[str],
    ) -> TransitionLookup:
        for field in item_fields:
            selected = [
                transition
                for transition in candidates
                if _canonical_action_mapping(transition.action).get(field)
                == proposed_key.get(field)
            ]
            if len(selected) == 1:
                transition = selected[0]
                return TransitionLookup(
                    status=TransitionLookupStatus.MATCH,
                    transition=transition,
                    target_state_id=transition.target,
                )
        return self._template_binding_missing_lookup()

    def resolve_transition(self, source_state_id: str, action: dict[str, Any]) -> TransitionLookup:
        """Resolve an action to a transition, preserving ambiguity as UNCERTAIN."""
        if source_state_id not in self.graph:
            return TransitionLookup(
                status=TransitionLookupStatus.NO_MATCH,
                details=f"State {source_state_id} is not in the FSM",
            )

        proposed_key = _canonical_action_mapping(action)
        proposed_type = proposed_key.get("type")
        same_type = [
            transition
            for transition in self.transitions
            if transition.source == source_state_id
            and _canonical_action_mapping(transition.action).get("type") == proposed_type
        ]
        proposed_fields = _transition_identity_fields(proposed_type, proposed_key)
        is_global = proposed_type in _GLOBAL_ACTION_TYPES
        state = self.states.get(source_state_id)
        template = None
        template_state_ids: set[str] = set()
        is_template_click = False
        item_fields: set[str] = set()
        has_template_entry_edge = False
        if (
            state
            and state.abstraction.template_id
            and state.abstraction.container_type == ContainerType.DYNAMIC
            and proposed_type == "click"
        ):
            template = self.sub_fsm_templates.get(state.abstraction.template_id)
            if template is not None:
                template_state_ids = set(template.states)
                # A click on this source is a template-binding attempt only if
                # the source actually has at least one outgoing click edge
                # *that is not a self-loop* whose target is inside the template
                # subgraph. Otherwise the click is chrome (toolbar Navigate up,
                # switch, etc.). Self-loops never count as template-entry
                # edges — entering a template item means leaving the source.
                has_template_entry_edge = any(
                    transition.target in template_state_ids and transition.target != source_state_id
                    for transition in same_type
                )
                is_template_click = has_template_entry_edge
                if is_template_click:
                    item_fields = self._item_specific_identity_fields(proposed_key)

        # On dynamic-template sources, resolve exact chrome clicks before
        # template-binding fallback. On ordinary states, the no-identity
        # ambiguity guard below must run first.
        exact_matches = [
            transition
            for transition in same_type
            if canonical_action_key(transition.action) == canonical_action_key(action)
        ]
        if is_template_click or source_state_id in template_state_ids:
            nontemplate_exact = [
                transition
                for transition in exact_matches
                if transition.target not in template_state_ids
            ]
            if len(nontemplate_exact) == 1:
                transition = nontemplate_exact[0]
                return TransitionLookup(
                    status=TransitionLookupStatus.MATCH,
                    transition=transition,
                    target_state_id=transition.target,
                )

        if is_template_click and not item_fields:
            return self._template_binding_missing_lookup()

        if proposed_type in _VALUE_PAYLOAD_ACTION_TYPES and not proposed_fields and same_type:
            return TransitionLookup(
                status=TransitionLookupStatus.UNCERTAIN,
                details=(
                    f"Action type {proposed_type!r} lacks target identity; "
                    "cannot bind it to a value-bearing transition"
                ),
            )

        if not proposed_fields and len(same_type) > 1 and not is_global:
            return TransitionLookup(
                status=TransitionLookupStatus.UNCERTAIN,
                details=(
                    f"Action type {proposed_type!r} lacks target identity; "
                    f"{len(same_type)} outgoing transitions share that type"
                ),
            )

        if len(exact_matches) == 1:
            transition = exact_matches[0]
            if is_template_click and transition.target in template_state_ids and not item_fields:
                return self._template_binding_missing_lookup()
            return TransitionLookup(
                status=TransitionLookupStatus.MATCH,
                transition=transition,
                target_state_id=transition.target,
            )
        if len(exact_matches) > 1:
            if is_template_click and any(
                transition.target in template_state_ids for transition in exact_matches
            ):
                return self._template_conflict_lookup(exact_matches, proposed_key, item_fields)
            return TransitionLookup(
                status=TransitionLookupStatus.UNCERTAIN,
                details="Multiple transitions share the same canonical action key",
            )

        if not proposed_fields and same_type and not is_global:
            return TransitionLookup(
                status=TransitionLookupStatus.UNCERTAIN,
                details=(
                    f"Action type {proposed_type!r} lacks target identity; "
                    "cannot bind it to a non-global transition"
                ),
            )

        if proposed_fields:
            compatible = [
                transition
                for transition in same_type
                if self._compatible_with_proposed_identity(
                    _canonical_action_mapping(transition.action), proposed_key, proposed_fields
                )
            ]
            if len(compatible) == 1:
                transition = compatible[0]
                if (
                    is_template_click
                    and transition.target in template_state_ids
                    and not item_fields
                ):
                    return self._template_binding_missing_lookup()
                return TransitionLookup(
                    status=TransitionLookupStatus.MATCH,
                    transition=transition,
                    target_state_id=transition.target,
                )
            if len(compatible) > 1:
                if is_template_click and any(
                    transition.target in template_state_ids for transition in compatible
                ):
                    return self._template_conflict_lookup(compatible, proposed_key, item_fields)
                return TransitionLookup(
                    status=TransitionLookupStatus.UNCERTAIN,
                    details=(
                        f"Action identity for type {proposed_type!r} matches "
                        f"{len(compatible)} outgoing transitions"
                    ),
                )

        if is_template_click and template is not None:
            # Scan for a concrete outgoing edge from this container to any
            # state inside the template subgraph whose canonical action
            # identity is compatible with the proposal. Identity-compatible
            # means every proposed identity field equals the stored one.
            identity_compatible_edges = [
                transition
                for transition in self.transitions
                if transition.source == source_state_id
                and transition.target in template.states
                and _canonical_action_mapping(transition.action).get("type") == proposed_type
                and self._compatible_with_proposed_identity(
                    _canonical_action_mapping(transition.action), proposed_key, proposed_fields
                )
            ]
            if len(identity_compatible_edges) == 1:
                transition = identity_compatible_edges[0]
                return TransitionLookup(
                    status=TransitionLookupStatus.MATCH,
                    transition=transition,
                    target_state_id=transition.target,
                )
            if len(identity_compatible_edges) > 1:
                return self._template_conflict_lookup(
                    identity_compatible_edges, proposed_key, item_fields
                )
            # No identity-compatible concrete edge exists; the action
            # claims to bind to the template but the FSM has no record
            # of an item-level edge with that identity.
            return self._template_binding_missing_lookup()

        return TransitionLookup(
            status=TransitionLookupStatus.NO_MATCH,
            details=f"No transition from {source_state_id} matches action {action}",
        )

    def is_valid_transition(self, source_state_id: str, action: dict[str, Any]) -> bool | None:
        """Check whether the model supports an action from the source state.

        For DYNAMIC container states with a sub_fsm_template, click actions are
        validated against the template's entry transition pattern rather than
        exact graph edges; a click that lacks the item-identity fields needed to
        select one transition resolves to None (UNCERTAIN), not True.

        Returns True for a resolved transition, False for a proven miss, and
        None when the action identity is insufficient to choose one transition.
        """
        lookup = self.resolve_transition(source_state_id, action)
        if lookup.status is TransitionLookupStatus.MATCH:
            return True
        if lookup.status is TransitionLookupStatus.UNCERTAIN:
            return None
        return False

    def is_reachable(self, source_state_id: str, goal_state_id: str) -> bool:
        """Check if goal_state_id is reachable from source_state_id. O(V+E) via BFS."""
        try:
            return nx.has_path(self.graph, source_state_id, goal_state_id)
        except nx.NodeNotFound:
            return False

    def get_shortest_path(self, source_state_id: str, goal_state_id: str) -> list[str]:
        """Get the shortest path from source_state_id to goal_state_id."""
        try:
            return nx.shortest_path(self.graph, source_state_id, goal_state_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_transition_target(self, source_state_id: str, action: dict[str, Any]) -> str | None:
        """Get the target state for a given action from a state."""
        lookup = self.resolve_transition(source_state_id, action)
        return lookup.target_state_id if lookup.status is TransitionLookupStatus.MATCH else None

    def get_transition(self, source_state_id: str, action: dict[str, Any]) -> Transition | None:
        """Get the Transition object for a given action from a state."""
        lookup = self.resolve_transition(source_state_id, action)
        return lookup.transition if lookup.status is TransitionLookupStatus.MATCH else None

    def find_state_by_fingerprint(self, fingerprint: str) -> str | None:
        """Find a state's exact functional-hash match for conservative evolution."""
        for state in self.states.values():
            if state.identity.functional_hash == fingerprint:
                return state.state_id
        return None

    def _serialize_payload(self) -> dict[str, Any]:
        """Build the canonical current-schema JSON payload shared by both writers."""
        return {
            "app_package": self.app_package,
            "efsm_revision": self.efsm_revision,
            "schema_version": SCHEMA_VERSION,
            "initial_state": self.initial_state,
            "states": {
                state_id: state.model_dump(mode="json") for state_id, state in self.states.items()
            },
            "transitions": [transition.model_dump() for transition in self.transitions],
            "construction_log": self.construction_log,
            "evolution_log": self.evolution_log,
            "sub_fsm_templates": {
                template_id: template.model_dump(mode="json")
                for template_id, template in self.sub_fsm_templates.items()
            },
        }

    def serialize(self, path: str | Path) -> None:
        """Serialize the FSM to current-schema JSON (nested-only canonical shape).

        Schema v6 writes each state as its canonical nested submodels (``identity``,
        ``android_context``, ``evidence``, ``abstraction``, ``invariant_specs``,
        ``annotations``) with no flat mirrors, and carries no scalar confidence axis.
        The reader also accepts v5 and migrates v4 payloads in memory.

        Delegates to :meth:`serialize_atomic`, so every write is crash-safe and there
        is no separate non-atomic write path that could leave a truncated file.
        """
        self.serialize_atomic(path)

    def serialize_atomic(self, path: str | Path) -> None:
        """Serialize atomically: write a same-directory temp file then ``os.replace``.

        Produces byte-for-byte the same current-schema content as :meth:`serialize`, but a
        crash mid-write can never leave a truncated ``fsm.json`` — the destination is
        either the old file or the fully written new one. Deliberately minimal: no
        locks, retries, backups, ``fsync`` protocols, or rollback framework.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Keep generated JSON readable in ordinary text editors.
        payload = json.dumps(self._serialize_payload(), indent=2, default=str) + "\n"
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(payload)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    @classmethod
    def deserialize(cls, path: str | Path) -> AppFSM:
        """Deserialize an FSM from a JSON file.

        Accepts any version in ``SUPPORTED_SCHEMA_VERSIONS``. A v4 payload is upgraded
        in memory by :func:`migrate_v4_payload`; the file itself is left untouched.
        State keys removed from the model are dropped silently for every accepted version.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        schema_version = data.get("schema_version")
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported FSM schema_version {schema_version!r}; "
                f"expected one of {SUPPORTED_SCHEMA_VERSIONS}"
            )
        if schema_version == "4":
            data = migrate_v4_payload(data)
        app_model = cls(app_package=data["app_package"])
        app_model.efsm_revision = int(data.get("efsm_revision", 0))
        app_model.initial_state = data.get("initial_state")
        app_model.construction_log = list(data.get("construction_log", []))
        app_model.evolution_log = list(data.get("evolution_log", []))

        def _state_from(state_data: dict[str, Any]) -> AbstractState:
            # Silently ignore state keys removed from the model.
            for removed in _REMOVED_STATE_KEYS:
                state_data.pop(removed, None)
            return AbstractState(**state_data)

        for state_data in data.get("states", {}).values():
            app_model.add_state(_state_from(state_data))

        # Top-level and template edges both carry the legacy ``low_trust`` spelling on
        # older bundles; ``Transition``'s own before-validator renames it, so neither
        # loop needs to know about the migration.
        for transition_data in data.get("transitions", []):
            app_model.add_transition(Transition(**transition_data))

        for template_data in data.get("sub_fsm_templates", {}).values():
            template_data["states"] = {
                state_id: _state_from(state)
                for state_id, state in template_data.get("states", {}).items()
            }
            template = SubFsmTemplate(**template_data)
            app_model.sub_fsm_templates[template.template_id] = template

        return app_model

    def __repr__(self) -> str:
        return (
            f"AppFSM(app={self.app_package!r}, "
            f"states={len(self.states)}, "
            f"transitions={len(self.transitions)})"
        )

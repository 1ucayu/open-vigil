"""M3A <-> Vigil interface glue and JSON bridge (the single glue file).

This module is the *only* place M3A action shapes are converted to Vigil
proposed actions, runtime XML is turned into a ``RawScreen``, and planner/
execution verdicts are formatted. It is invoked by the M3A subprocess as::

    <repo_python> -m vigil.integration.m3a_adapter

reading exactly one JSON request object from stdin and writing exactly one JSON
response object to stdout (diagnostics go to stderr). The main Vigil process may
import this module directly; it never imports ``android_world``. The M3A
subprocess never imports Vigil — it only shells out to this bridge.

The same module also serves the persistent transport::

    <repo_python> -m vigil.integration.m3a_adapter --serve

which reads newline-delimited JSON requests and writes exactly one
newline-delimited JSON response per request, reusing one interpreter, one import
graph, and one parsed ``AppFSM`` across calls. Both transports dispatch through
the identical :func:`handle_request`, so a successful response is byte-identical
either way; only the per-call process cost differs.

Boundary: requests carry only public data (mode, fsm_path, app_package,
screen_xml, action, planner ``intent_patch``, and Vigil's admitted planner
proposal). ``goal_state``
stays ``None`` at this boundary — no oracle-derived expected state, hidden intent, or
expected verdict is ever accepted.

Intervention modes:

- ``planner``: soft advisory over the legal outgoing actions from the localized
  FSM state. Never executes or blocks.
- ``intent_patch``: planner-boundary admission of the incremental patch emitted
  beside M3A's action proposal. It binds that opaque proposal to the resulting
  snapshot revision and never reads the screen or calls an LLM.
- ``execution``: pre-action verification of a concrete proposed action against
  the live screen. ``should_execute`` is true only for an ALLOW verdict; every
  other outcome (DENY or UNCERTAIN) blocks execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

from pydantic import BaseModel, ConfigDict, ValidationError

from vigil.core.config import LLMConfig
from vigil.core.llm_client import LlmClient
from vigil.integration.runtime_evidence import (
    load_persisted_runtime_screens,
    load_runtime_evidence,
    normalize_trace_screen,
    persist_runtime_screen,
)
from vigil.models.evolution import EvolutionOutcome
from vigil.models.fsm import (
    AppFSM,
    HierarchyLevel,
    Transition,
    TransitionLookupStatus,
    canonical_action_key,
    canonical_transition_action,
)
from vigil.models.guard import (
    GuardAdmissionStatus,
    GuardKind,
    IntentSlot,
    SlotRequiredness,
)
from vigil.models.intent_binding import (
    AdmittedPlannerProposal,
    IntentPatch,
    RevisionedIntentBindingSnapshot,
)
from vigil.models.state import RawScreen
from vigil.neuro.evolution import RuntimeEvolutionEngine, SemanticGapEvidencePacket
from vigil.runtime.action_binder import ActionBinder, BindingStatus
from vigil.runtime.continuation import (
    ContinuationEligibility,
    ContinuationRefusal,
    PostActionWitness,
    SemanticGapContinuationClassifier,
    TopologyAuthorization,
    TopologyContinuationClassifier,
    classify_post_action_witness,
)
from vigil.runtime.monitor import RuntimeMonitor, RuntimeVerificationResult
from vigil.runtime.observed_transition import apply_observed_transition
from vigil.runtime.screen_capture import raw_screen_from_xml
from vigil.symbolic.clause_activator import (
    ClauseActivator,
    ClauseDisposition,
)
from vigil.symbolic.decision_engine import DecisionEngine
from vigil.symbolic.dsl_evaluator import (
    DSLEvaluator,
    GuardStatus,
    IntentContext,
    PredicateWitness,
)
from vigil.symbolic.fsm_checker import (
    SemanticGuardCoverageGap,
    VerificationOutput,
    VerifyReason,
    VerifyResult,
    is_conservative_refusal,
    semantic_guard_coverage_gap,
)
from vigil.symbolic.intent_patch import (
    admit_intent_patch,
    canonical_agent_action_proposal,
    derive_intent_patch_demand,
)
from vigil.symbolic.intent_rebind import (
    RecoveryAuthorization,
    admit_intent_rebind,
    build_recovery_authorization,
)
from vigil.symbolic.planner_verifier import PlannerVerification, PlannerVerifier
from vigil.symbolic.state_locator import (
    StateLocator,
    is_known_location,
    raw_screen_canonical_equivalence_hash,
)

_MODES = (
    "planner",
    "intent_patch",
    "intent_rebind",
    "execution",
    "post_action",
    "semantic_evolution",
    "topology_evolution",
    "observe_witness",
)
_SEMANTIC_GAP_POLICIES = (
    "block",
    "evolve_before_execution",
    "execute_unverified_for_evolution",
)
# Policies that may authorize one unverified *topology* evidence execution. A topology
# gap has no pre-execution semantic repair, so both non-blocking policies treat it the
# same way; ``block`` still blocks.
_TOPOLOGY_CONTINUATION_POLICIES = frozenset(
    {"evolve_before_execution", "execute_unverified_for_evolution"}
)

# Fixed process constraints attached to every blocked observation. The agent
# must not assume the blocked action ran or changed the app state.
_PROCESS_CONSTRAINTS = (
    "The checked action was not executed.",
    "Do not assume the app state changed.",
    "Use the current screen, UI elements, task goal, and this verifier "
    "observation for the next step.",
)

_PLANNER_CONSTRAINTS = (
    "No Android action has run for this planner observation.",
    "This is current-state EFSM evidence, not an execution-time block.",
    "Execution-time Vigil still gates each concrete GUI action.",
)

# Fixed process constraints attached to every post-action observation. Unlike the
# blocked (pre-action) constraints, the checked action has already executed; the
# agent must reason from the landed screen forward, not assume nothing changed.
_POST_ACTION_PROCESS_CONSTRAINTS = (
    "The checked action has already executed; this is a post-action observation.",
    "Do not assume any further action has run beyond the one already executed.",
    "Use the current screen, UI elements, task goal, and this verifier "
    "observation to decide the next step.",
)


class SemanticGapContinuationReceipt(BaseModel):
    """Capability for recording one explicitly unverified evolution exposure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str
    binding_revision: int
    efsm_revision: int
    source_state_id: str
    target_state_id: str
    semantic_guard_gap: SemanticGuardCoverageGap
    bound_action: dict[str, Any]


class TopologyContinuationReceipt(BaseModel):
    """Capability for one explicitly unverified topology-evolution exposure.

    Issued only for an auto-eligible action on a conservatively refused topology gap
    (STATE_UNKNOWN / INCOMPLETE_COVERAGE). It carries the verifier outcome and the real
    foreground context observed at execution time. The post-execution
    ``topology_evolution`` mode revalidates the receipt, so admission binds the same
    proposal, action, and foreground that were authorized.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str
    binding_revision: int
    efsm_revision: int
    verifier_result: str
    verifier_reason: str
    authorization_basis: str
    source_state_id: str | None = None
    bound_action: dict[str, Any]
    foreground_package: str
    foreground_activity: str


class _MeasuredEvolutionLlm:
    """Measure one semantic-generation stage without changing the LLM client API."""

    def __init__(self, delegate: LlmClient) -> None:
        self._delegate = delegate
        self.call_count = 0
        self.latency_ms = 0.0

    def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        started = time.perf_counter()
        try:
            return self._delegate.generate_structured(*args, **kwargs)
        finally:
            self.latency_ms += (time.perf_counter() - started) * 1000.0

    def generate_structured_with_images(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        started = time.perf_counter()
        try:
            return self._delegate.generate_structured_with_images(*args, **kwargs)
        finally:
            self.latency_ms += (time.perf_counter() - started) * 1000.0


# --- action conversion (the only conversion policy) ----------------------


def _attach_target(proposed: dict[str, Any], element: dict[str, Any]) -> None:
    """Copy stable identity fields from the M3A element snapshot onto ``proposed``.

    Prefers a real ``resource_id`` then ``resource_name``; never fabricates a
    synthetic ``target``/``m3a_<index>`` handle. Runtime XML remains the source of
    truth for what actually resolves.

    ``target_bounds`` is carried through when the agent supplies it. It is not
    identity — ``fsm._ACTION_KEY_FIELDS`` and ``fsm._SELECTOR_IDENTITY_FIELDS``
    are closed allowlists that exclude it, the same way capture-local bounds and
    depth are already excluded — it is the only channel by which a proposal can
    name *which* of several identity-less rows it meant, since those rows produce
    byte-identical element snapshots. The binder confirms it against live XML and
    refuses on any mismatch.
    """
    resource_id = element.get("resource_id") or element.get("resource_name")
    if resource_id:
        proposed["target_resource_id"] = resource_id
    if element.get("text"):
        proposed["target_text"] = element["text"]
    if element.get("content_description"):
        proposed["target_content_desc"] = element["content_description"]
    if element.get("class_name"):
        proposed["target_class_name"] = element["class_name"]
    bounds = element.get("bounds")
    if (
        isinstance(bounds, list | tuple)
        and len(bounds) == 4
        and all(isinstance(v, int) for v in bounds)
    ):
        proposed["target_bounds"] = list(bounds)


def convert_m3a_action(m3a_action: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one M3A action into a Vigil proposed-action dict, or ``None``.

    ``None`` means the action has no Vigil executable form (unsupported, or a
    scroll direction Vigil does not model). Unsupported actions must never be
    silently allowed — the caller turns ``None`` into a blocking DENY.
    """
    action_type = m3a_action.get("action_type")
    element = m3a_action.get("element") or {}

    if action_type in ("navigate_back", "navigate_home"):
        return {"type": action_type}

    if action_type in ("click", "long_press", "input_text"):
        proposed: dict[str, Any] = {"type": action_type}
        _attach_target(proposed, element)
        if action_type == "input_text":
            proposed["text"] = m3a_action.get("text")
        return proposed

    if action_type == "scroll":
        direction = (m3a_action.get("direction") or "").lower()
        mapped = {"up": "scroll_up", "down": "scroll_down"}.get(direction)
        if mapped is None:
            return None
        proposed = {"type": mapped}
        _attach_target(proposed, element)
        return proposed

    # status, answer, wait, open_app, keyboard_enter, coordinate-only, unknown.
    return None


# --- screen + verdict formatting -----------------------------------------


def _infer_activity_name(fsm: AppFSM) -> str | None:
    activities = {
        state.android_context.activity_name
        for state in fsm.states.values()
        if state.android_context.activity_name
    }
    if len(activities) == 1:
        return next(iter(activities))
    return None


def _verdict(result: VerifyResult) -> str:
    return result.value.upper()


def _live_element_summary(element: dict[str, Any]) -> dict[str, Any]:
    """Prompt-facing identity for a live M3A UI element."""
    return {
        key: value
        for key, value in {
            "index": element.get("index"),
            "resource_id": element.get("resource_id"),
            "resource_name": element.get("resource_name"),
            "text": element.get("text"),
            "content_description": element.get("content_description"),
            "class_name": element.get("class_name"),
            "is_clickable": element.get("is_clickable"),
            "is_long_clickable": element.get("is_long_clickable"),
            "is_editable": element.get("is_editable"),
            "is_scrollable": element.get("is_scrollable"),
            "is_enabled": element.get("is_enabled"),
        }.items()
        if value not in (None, "")
    }


def _live_element_supports_action(action_type: str | None, element: dict[str, Any]) -> bool:
    if action_type == "click":
        return element.get("is_clickable") is True
    if action_type == "long_press":
        return element.get("is_long_clickable") is True
    if action_type == "input_text":
        return element.get("is_editable") is True
    if action_type in ("scroll_up", "scroll_down"):
        return element.get("is_scrollable") is True
    return True


def _live_action_match(
    action: dict[str, Any], live_ui_elements: list[dict[str, Any]] | None
) -> dict[str, Any] | None:
    """Resolve an FSM action identity to the current M3A UI index, if exact."""
    if not live_ui_elements:
        return None
    action_type = action.get("type") or action.get("action_type")
    target_resource_id = action.get("target_resource_id") or action.get("resource_id")
    target_text = action.get("target_text")
    target_content_desc = action.get("target_content_desc")

    for element in live_ui_elements:
        if target_resource_id and target_resource_id not in (
            element.get("resource_id"),
            element.get("resource_name"),
        ):
            continue
        if not target_resource_id and target_text and target_text != element.get("text"):
            continue
        if (
            not target_resource_id
            and not target_text
            and target_content_desc
            and target_content_desc != element.get("content_description")
        ):
            continue
        if not target_resource_id and not target_text and not target_content_desc:
            continue
        if _live_element_supports_action(action_type, element):
            return _live_element_summary(element)
    return None


def _with_live_action_match(
    action_summary: dict[str, Any],
    action: dict[str, Any],
    live_ui_elements: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    match = _live_action_match(action, live_ui_elements)
    if match is not None:
        action_summary["live_ui_index"] = match["index"]
        action_summary["live_ui_match"] = match
    return action_summary


def _action_descriptor(
    action: dict[str, Any],
    target_state: str,
    live_ui_elements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compact, prompt-friendly summary of one outgoing transition."""
    visible_action = canonical_transition_action(action)
    descriptor = {
        "action_type": visible_action.get("type"),
        "target_text": visible_action.get("target_text"),
        "target_resource_id": visible_action.get("target_resource_id")
        or visible_action.get("resource_id"),
        "target_content_desc": visible_action.get("target_content_desc"),
        "target_state": target_state,
    }
    return _with_live_action_match(descriptor, visible_action, live_ui_elements)


def _descriptor_label(descriptor: dict[str, Any]) -> str:
    target = (
        descriptor.get("target_text")
        or descriptor.get("target_resource_id")
        or descriptor.get("target_content_desc")
        or "global"
    )
    label = f"{descriptor.get('action_type')}({target})"
    if descriptor.get("live_ui_index") is not None:
        label += f" -> UI index {descriptor['live_ui_index']}"
    return label


# Commit-gated guard kinds: DecisionEngine refuses these
# (``COMMIT_INTENT_UNBOUND``) while every intent-bound clause is skipped.
_COMMIT_GUARD_KINDS = frozenset({GuardKind.CONFIRM_COMMIT, GuardKind.SAFETY_CHECK})


def _transition_readiness(
    transition: Any,
    transition_guide: dict[str, Any],
    selectable: bool,
) -> tuple[bool, str]:
    """Return one transition's planner-time readiness flag and blocking note.

    ``ready`` is structural selectability plus the absence of a missing required
    intent slot. The note names the concrete execution-time block that a
    structurally legal action would still hit, reusing the activation that
    :func:`_with_clause_activation_guidance` already computed:

    - a required slot that is not bound makes its clause inconclusive;
    - a commit/safety contract whose intent-bound clauses are all unbound is held
      by the DecisionEngine commit gate.

    Missing-required is reported first because clause activation is checked before
    the commit gate at runtime.
    """
    missing_required = list(transition_guide["missing_required_intent_slots"])
    ready = selectable and not missing_required
    if missing_required:
        names = ", ".join(missing_required)
        return ready, f"[structurally legal; will block until intent slot {names} is bound]"

    contract = transition.guard_contract
    if contract is not None and contract.kind in _COMMIT_GUARD_KINDS:
        active = list(transition_guide["active_intent_slots"])
        intent_bound = sorted(
            set(active)
            | set(transition_guide["skipped_optional_intent_slots"])
            | set(transition_guide["inconclusive_intent_slots"])
        )
        if intent_bound and not active:
            # The commit gate will refuse this transition, so an honest
            # readiness flag reports it as not ready even though it is
            # structurally selectable and no single slot is individually
            # required.
            names = ", ".join(intent_bound)
            return False, f"[will block unless at least one intent slot of: {names} is bound]"
    return ready, ""


# --- shared guard-predicate evidence helpers -----------------------------


def _guard_predicate_from_witness(
    witness: PredicateWitness,
    trace: Any | None,
) -> dict[str, Any]:
    """Build one ``guard_predicate`` evidence item from a real evaluator witness.

    The operands and per-predicate result come from the exact pass that produced
    the verdict — never a re-parse or re-read. ``trace`` is the ACTIVE clause this
    predicate belongs to (for intent-slot attribution); ``None`` when the
    predicate could not be positionally attributed.
    """
    item: dict[str, Any] = {
        "kind": "guard_predicate",
        "predicate_type": witness.predicate_type,
        "expression": trace.compiled_expr if trace is not None else None,
        "operator": witness.operator or None,
        "status": witness.result.value,
    }
    if witness.predicate_type == "action":
        left: dict[str, Any] = {"source": "action", "property": witness.property}
    else:
        left = {"source": "screen", "selector": witness.element}
        prop = witness.property if witness.predicate_type == "read" else "value"
        if prop is not None:
            left["property"] = prop
    if witness.actual is not None:
        left["value"] = witness.actual
    item["left_value"] = left

    slot = trace.slot if trace is not None else None
    if slot is not None:
        right: dict[str, Any] = {"source": "intent", "slot": slot}
    else:
        right = {"source": "literal"}
    if witness.expected is not None:
        right["value"] = witness.expected
    item["right_value"] = right
    return item


def _witness_predicate_checks(
    witness: tuple[PredicateWitness, ...] | None,
    activation: Any | None,
    *,
    preview: bool = False,
) -> list[dict[str, Any]]:
    """Attribute real-pass predicate witnesses to their ACTIVE clauses in order.

    One predicate per admitted clause is the standing invariant; when the counts
    match, ``witness[i]`` is attributed to the i-th ACTIVE clause. On any
    cardinality mismatch, the witnesses are still reported without slot
    attribution rather than guessing.
    """
    records = list(witness or ())
    active = (
        [clause for clause in activation.clauses if clause.disposition is ClauseDisposition.ACTIVE]
        if activation is not None
        else []
    )
    aligned = len(records) == len(active)
    items: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        trace = active[index] if aligned else None
        item = _guard_predicate_from_witness(record, trace)
        if preview:
            item["preview"] = True
        items.append(item)
    return items


def _refuted_predicates(
    witness: tuple[PredicateWitness, ...] | None,
    activation: Any | None,
) -> list[dict[str, Any]]:
    """Compact witness operands for every predicate the real pass proved FALSE.

    This is a projection of the ``guard_predicate`` evidence items — same
    evaluator pass, same operands, no re-read — flattened so the agent can see
    *what the screen said* next to *what the frozen binding required* without
    walking the nested evidence tree. ``slot`` is the ACTIVE clause's intent slot
    when the predicate could be positionally attributed, so the observation
    renderer can name the binding rather than only the element.
    """
    records = list(witness or ())
    active = (
        [clause for clause in activation.clauses if clause.disposition is ClauseDisposition.ACTIVE]
        if activation is not None
        else []
    )
    aligned = len(records) == len(active)
    refuted: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if record.result is not GuardStatus.FALSE:
            continue
        trace = active[index] if aligned else None
        refuted.append(
            {
                "element": record.element,
                "property": record.property,
                "operator": record.operator or None,
                "expected": record.expected,
                "actual": record.actual,
                "slot": trace.slot if trace is not None else None,
            }
        )
    return refuted


# Echo operators: a FALSE witness under one of these, against a *screen* surface,
# is an attestation mismatch — the live literal is the canonical candidate for the
# slot. Any other shape (action predicate, constraint operator) means the value must
# still be re-derived from the user instruction, matching the ``evidence_surface``
# roles in ``vigil.symbolic.intent_patch``.
_ECHO_OPERATORS = frozenset({"=="})
_SCREEN_PREDICATE_TYPES = frozenset({"read", "value"})


def _screen_attested_slots(
    witness: tuple[PredicateWitness, ...] | None,
    activation: Any | None,
) -> set[str]:
    """Slot names whose FALSE witness is a screen-surface echo comparison."""
    records = list(witness or ())
    active = (
        [clause for clause in activation.clauses if clause.disposition is ClauseDisposition.ACTIVE]
        if activation is not None
        else []
    )
    if len(records) != len(active):
        return set()
    attested: set[str] = set()
    for record, trace in zip(records, active, strict=True):
        if trace.slot is None or record.result is not GuardStatus.FALSE:
            continue
        if record.predicate_type in _SCREEN_PREDICATE_TYPES and record.operator in _ECHO_OPERATORS:
            attested.add(trace.slot)
    return attested


def _rebind_affordance(
    authorization: RecoveryAuthorization,
    witness: tuple[PredicateWitness, ...] | None,
    activation: Any | None,
) -> dict[str, Any]:
    """Presentation-only view of an issued recovery authorization.

    The authorization itself stays harness control data; this is the part the
    observation section may say out loud. ``value_source`` mirrors the
    ``evidence_surface`` role rules: only an attestation-shaped refutation makes
    the observed screen literal the canonical candidate, so the rendered prose can
    never contradict the per-round binding rules for an instruction slot.
    """
    attested = _screen_attested_slots(witness, activation)
    return {
        "slots": [
            {
                "name": slot.name,
                "observed_value": slot.observed_value,
                "value_source": ("screen_literal" if slot.name in attested else "user_instruction"),
            }
            for slot in authorization.eligible_slots
        ]
    }


def _admitted_clause_expressions(transition: Any) -> list[str]:
    contract = transition.guard_contract if transition is not None else None
    if contract is None or contract.admission_status is not GuardAdmissionStatus.ADMITTED:
        return []
    return [
        clause.compiled_expr
        for clause in contract.clauses
        if clause.admission_status is GuardAdmissionStatus.ADMITTED and clause.compiled_expr
    ]


def _guard_check(transition: Any, status: str, details: str | None = None) -> dict[str, Any]:
    check: dict[str, Any] = {
        "kind": "guard",
        "expressions": _admitted_clause_expressions(transition),
        "guard_admission_status": _guard_status(transition) if transition is not None else None,
        "status": status,
    }
    if details is not None:
        check["details"] = details
    return check


# --- agent-visible EFSM guide --------------------------------------------

# Builder fallback state name: the raw window/resource identity plus an optional
# disambiguating ordinal, e.g. "@7F090001 #3". Not app-specific — it is the shape
# ``FsmBuilder._derive_state_name`` emits when no readable page identity exists.
_AUTO_LABEL_PATTERN = re.compile(r"@[0-9A-Fa-f]+\s*(#\d+)?")


def _state_label(fsm: AppFSM, state_id: str | None) -> str:
    if state_id is None:
        return ""
    state = fsm.states.get(state_id)
    if state is None:
        return state_id
    display_name = state.annotations.display_name
    return display_name or state.name or state.state_id


def _state_summary(fsm: AppFSM, state_id: str | None) -> dict[str, Any] | None:
    if state_id is None:
        return None
    state = fsm.states.get(state_id)
    if state is None:
        return {"state_id": state_id, "label": state_id}
    return {
        "state_id": state.state_id,
        "label": _state_label(fsm, state.state_id),
        "kind": "dialog" if state.hierarchy_level == HierarchyLevel.COMPONENT else "normal",
        "hierarchy_level": state.hierarchy_level.value,
        "container_type": state.abstraction.container_type.value,
        "page_function": state.annotations.page_function,
    }


def _action_summary(
    action: dict[str, Any],
    live_ui_elements: list[dict[str, Any]] | None = None,
    *,
    include_value_payload: bool = False,
) -> dict[str, Any]:
    """Stable action identity fields exposed to the mobile agent."""
    visible_action = action if include_value_payload else canonical_transition_action(action)
    summary = {
        key: value
        for key, value in {
            "type": visible_action.get("type") or visible_action.get("action_type"),
            "target": visible_action.get("target"),
            "target_resource_id": visible_action.get("target_resource_id")
            or visible_action.get("resource_id"),
            "target_text": visible_action.get("target_text"),
            "target_content_desc": visible_action.get("target_content_desc"),
            "input_text": visible_action.get("text") or visible_action.get("value"),
        }.items()
        if value not in (None, "")
    }
    return _with_live_action_match(summary, visible_action, live_ui_elements)


def _slot_from_contract(slot: IntentSlot) -> dict[str, Any]:
    return {
        "name": slot.name,
        "type": slot.slot_type.value,
        "requiredness": slot.requiredness.value,
        "description": slot.description,
        "value_domain": list(slot.value_domain),
    }


def _transition_slots(transition: Transition) -> list[dict[str, Any]]:
    contract = transition.guard_contract
    if contract is None or contract.admission_status is not GuardAdmissionStatus.ADMITTED:
        return []
    transition_usage = {
        "source": transition.source,
        "target": transition.target,
        "guard_clauses": _admitted_clause_expressions(transition),
        "action_type": transition.action.get("type") or transition.action.get("action_type"),
    }
    slots = []
    for slot in contract.slots:
        slot_spec = _slot_from_contract(slot)
        slot_spec["transition_usage"] = [transition_usage]
        slots.append(slot_spec)
    return sorted(slots, key=lambda slot: slot["name"])


def _invariant_summaries(fsm: AppFSM, state_id: str | None) -> list[dict[str, Any]]:
    if state_id is None or state_id not in fsm.states:
        return []
    return [
        {
            "expr": invariant.expr,
            "source": invariant.source,
            "evidence_count": invariant.evidence_count,
        }
        for invariant in fsm.states[state_id].invariant_specs
    ]


def _guard_status(transition: Transition) -> str | None:
    contract = transition.guard_contract
    return contract.admission_status.value if contract is not None else None


def _transition_guide(
    fsm: AppFSM,
    transition: Any,
    selectable: bool | None = None,
    live_ui_elements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    clauses = _admitted_clause_expressions(transition)
    guide = {
        "from": transition.source,
        "to": transition.target,
        "action": _action_summary(transition.action, live_ui_elements),
        "guard_clauses": clauses,
        "intent_slots": _transition_slots(transition),
        "guard_admission_status": _guard_status(transition),
    }
    if selectable is not None:
        guide["selectable"] = selectable
    guide["target_invariants"] = _invariant_summaries(fsm, transition.target)
    return guide


def _binding_snapshot_policy() -> dict[str, Any]:
    """Canonical runtime binding protocol owned by the mobile agent."""
    return {
        "owner": "mobile_agent",
        "snapshot_shape": {
            "revision": "non-negative integer",
            "values": {"<slot_name>": "<JSON scalar>"},
            "slot_fingerprints": {"<slot_name>": "<Vigil-owned fingerprint>"},
        },
        "planner_patch_shape": {
            "base_revision": "current snapshot revision",
            "updates": {"<demanded_slot>": "<JSON scalar>"},
        },
        # The per-round intent-binding section is the authoritative statement of the
        # patch rules (presence, omission, add-only freezing, revision behavior). This
        # once-per-episode guide points at it instead of restating it, so the two
        # sections cannot drift apart in the same prompt.
        "binding_rules": (
            "The per-round Vigil intent-binding section states the authoritative patch "
            "rules — when a slot counts as bound, when to omit it, which keys are frozen, "
            "and how the revision moves. Follow that section; this guide does not restate it."
        ),
        "number_value": "A slot declared as number must carry a JSON number, not numeric text.",
        "activation": {
            "no_intent_slot": "always_active",
            "slot_present": "active",
            "required+missing": "inconclusive",
            "optional+missing": "skipped",
        },
    }


def _coerce_binding_snapshot(
    binding_snapshot: RevisionedIntentBindingSnapshot | dict[str, Any],
) -> RevisionedIntentBindingSnapshot:
    """Parse the revisioned snapshot exchanged at the M3A boundary."""
    return RevisionedIntentBindingSnapshot.model_validate(binding_snapshot)


def _coerce_admitted_proposal(
    proposal: AdmittedPlannerProposal | dict[str, Any],
) -> AdmittedPlannerProposal:
    return AdmittedPlannerProposal.model_validate(proposal)


def _opaque_m3a_action_proposal(action: dict[str, Any]) -> dict[str, Any]:
    """Remove execution-only live element evidence from an M3A proposal."""
    return {key: value for key, value in action.items() if key != "element" and value is not None}


def _planner_selectability(
    fsm: AppFSM,
    state_id: str,
    outgoing: list[Transition],
) -> tuple[list[bool], PlannerVerification]:
    """Return the canonical goal-free PlannerVerifier mask for one state."""
    verification = PlannerVerifier(fsm).verify_candidates(
        state_id,
        [transition.action for transition in outgoing],
        goal_state=None,
    )
    return list(verification.action_mask), verification


def _with_clause_activation_guidance(
    guidance: dict[str, Any],
    outgoing: list[Any],
    binding_snapshot: RevisionedIntentBindingSnapshot,
) -> dict[str, Any]:
    """Annotate planner guidance with the same activation result used at runtime."""
    updated = dict(guidance)
    transition_guides = [dict(item) for item in guidance["outgoing_transitions"]]
    activator = ClauseActivator()

    for transition, transition_guide in zip(outgoing, transition_guides, strict=True):
        contract = transition.guard_contract
        activation = (
            activator.activate(contract, binding_snapshot.values)
            if contract is not None and contract.admission_status is GuardAdmissionStatus.ADMITTED
            else None
        )
        traces = activation.clauses if activation is not None else []
        missing_required = {
            trace.slot
            for trace in traces
            if trace.slot is not None
            and trace.requiredness is SlotRequiredness.REQUIRED
            and trace.slot_bound is False
        }
        inconclusive_required = {
            trace.slot
            for trace in traces
            if trace.slot is not None
            and trace.requiredness is SlotRequiredness.REQUIRED
            and trace.disposition is ClauseDisposition.INCONCLUSIVE
        }
        inconclusive = {
            trace.slot
            for trace in traces
            if trace.slot is not None and trace.disposition is ClauseDisposition.INCONCLUSIVE
        }
        active = {
            trace.slot
            for trace in traces
            if trace.slot is not None and trace.disposition is ClauseDisposition.ACTIVE
        }
        skipped = {
            trace.slot
            for trace in traces
            if trace.slot is not None and trace.disposition is ClauseDisposition.SKIPPED
        }
        transition_guide["clause_activation"] = (
            activation.model_dump(mode="json") if activation is not None else None
        )
        transition_guide["active_intent_slots"] = sorted(active)
        transition_guide["skipped_optional_intent_slots"] = sorted(skipped)
        transition_guide["missing_required_intent_slots"] = sorted(missing_required)
        transition_guide["inconclusive_required_intent_slots"] = sorted(inconclusive_required)
        transition_guide["inconclusive_intent_slots"] = sorted(inconclusive)

    updated["outgoing_transitions"] = transition_guides
    return updated


def _is_raw_auto_label(label: str) -> bool:
    """True for a builder fallback name that carries no page identity.

    When no activity label, page title, heading, or interactable text is
    available, the FSM builder falls back to the raw window identity plus a
    disambiguating ordinal (``@7F090001 #3``). Such a name tells the planner
    nothing the ``state_id`` does not already say.
    """
    return _AUTO_LABEL_PATTERN.fullmatch(label.strip()) is not None


def _is_informative_state(summary: dict[str, Any]) -> bool:
    """Keep a guide state entry only when it names the screen for a human reader."""
    if summary.get("page_function"):
        return True
    return not _is_raw_auto_label(str(summary.get("label") or ""))


def build_agent_efsm_guide(fsm: AppFSM) -> dict[str, Any]:
    """Return an app-agnostic EFSM contract view for mobile-agent planning.

    This is derived only from the loaded FSM/guard metadata. It is a prompt-facing
    guide, not a verifier verdict source. States with neither a human label nor a
    ``page_function`` are omitted from ``states`` — their entry would be a raw
    window ordinal the planner cannot reason about — while ``topology`` still
    carries every edge, so no transition or state id disappears from the guide.
    """
    topology = []
    for transition in fsm.transitions:
        slots = _transition_slots(transition)
        topology.append(
            {
                "from": transition.source,
                "to": transition.target,
                "action": _action_summary(transition.action),
                "intent_slots": slots,
                "has_guard": bool(
                    transition.guard_contract
                    and transition.guard_contract.admission_status is GuardAdmissionStatus.ADMITTED
                    and _admitted_clause_expressions(transition)
                ),
                "target_has_invariants": bool(
                    fsm.states.get(transition.target)
                    and fsm.states[transition.target].invariant_specs
                ),
            }
        )
    summaries = [_state_summary(fsm, state_id) for state_id in sorted(fsm.states)]
    named_states = [
        summary for summary in summaries if summary is not None and _is_informative_state(summary)
    ]
    return {
        "guide_type": "agent_visible_efsm",
        "app_package": fsm.app_package,
        "efsm_revision": fsm.efsm_revision,
        "initial_state": fsm.initial_state,
        "states": named_states,
        "unlabeled_state_count": len(summaries) - len(named_states),
        "topology": topology,
        "binding_snapshot_policy": _binding_snapshot_policy(),
        "verifier_boundary": (
            "Use this app-derived EFSM guide for planning only; Vigil still makes "
            "the execution-time ALLOW/DENY verdict."
        ),
    }


# --- unlocalized-screen recovery guidance --------------------------------

# Explicit navigation actions are recovery affordances by type alone.
_RECOVERY_NAVIGATION_TYPES = frozenset({"navigate_back", "navigate_home"})

_UNLOCALIZED_STATEMENT = (
    "This screen is not in the app model; no proposal can be verified here and "
    "intent patches cannot release this block."
)
_UNLOCALIZED_ESCALATION = "Navigate back toward a modeled screen before retrying this action."
_RECOVERY_ACTIONS_CAVEAT = (
    "These are existing modeled edges out of the last localized state, not a plan and "
    "not a verified route from the current screen. The app has since moved off that "
    "state, so they may be stale; Vigil still verifies whatever action you choose."
)
_UNLOCALIZED_ROUNDS_ESCALATION_THRESHOLD = 2

# Cold start: the screen is unlocalized and no screen has been localized yet, so
# there is no anchor to reason from. The model's own entry screens are the only
# evidence available, and they are offered as evidence only -- naming them
# changes no trust flag and authorizes no action.
_COLD_START_STATEMENT = (
    "This screen is not in the app model and Vigil has not localized any screen yet "
    "in this episode, so there is no anchor to reason from. No proposal can be "
    "verified here and intent patches cannot release this block."
)
_ENTRY_CANDIDATES_CAVEAT = (
    "These are the app model's entry screens and the edges that already leave them, "
    "listed as evidence about the model — not a claim about the current screen, not a "
    "plan, and not authorization to act. Vigil has matched nothing here; if the current "
    "screen is a first-run or onboarding screen the model never observed, no action on "
    "it can be verified. Vigil still verifies whatever action you choose."
)
_COLD_START_ESCALATION = (
    "Vigil has still localized nothing this episode; the screen is most likely outside "
    "the app model rather than a wrong turn inside it."
)
#: Entry screens are the initial state plus its activity siblings. Both bounds keep
#: the payload readable; they truncate the listing, never the model.
_MAX_ENTRY_CANDIDATES = 4
_MAX_ENTRY_CANDIDATE_ACTIONS = 12


def _recovery_action_kind(fsm: AppFSM, transition: Any) -> str | None:
    """Classify one outgoing edge as a structural recovery affordance, or ``None``.

    Two shapes qualify, both read directly from the transition graph:

    - ``navigation``: an explicit ``navigate_back``/``navigate_home`` action, which
      moves toward a modeled screen by construction;
    - ``nav_tab``: a click onto a non-dialog state that has a modeled path back to
      the source. Reciprocity is what makes a tab a tab; a one-way click deeper
      into a flow is not a way out of a dead end.

    No resource id, label, or package string is inspected, so this stays
    app-agnostic.
    """
    action_type = transition.action.get("type") or transition.action.get("action_type")
    if action_type in _RECOVERY_NAVIGATION_TYPES:
        return "navigation"
    if action_type != "click":
        return None
    target = fsm.states.get(transition.target)
    if target is None or target.hierarchy_level is HierarchyLevel.COMPONENT:
        return None
    if transition.target == transition.source:
        return None
    if not fsm.is_reachable(transition.target, transition.source):
        return None
    return "nav_tab"


def _recovery_actions(fsm: AppFSM, state_id: str) -> list[dict[str, Any]]:
    """Structural ways out of ``state_id``, derived only from existing edges."""
    actions: list[dict[str, Any]] = []
    for transition in fsm.transitions:
        if transition.source != state_id:
            continue
        kind = _recovery_action_kind(fsm, transition)
        if kind is None:
            continue
        actions.append(
            {
                "kind": kind,
                "action": _action_summary(transition.action),
                "target_state_id": transition.target,
                "target_label": _state_label(fsm, transition.target),
            }
        )
    return actions


def _entry_candidate_state_ids(fsm: AppFSM) -> list[str]:
    """Modeled screens the app can be sitting on right after launch.

    Read only out of the model: the recorded initial state, then its
    activity siblings (other modeled screens of the launch activity), which is
    where a cold start lands when the app restores its own last screen. Dialog
    (component-level) states are excluded — a dialog is never an entry screen.
    No resource id, label, or package string is inspected, so this stays
    app-agnostic.
    """
    initial = fsm.initial_state
    if initial is None or initial not in fsm.states:
        return []
    activity = fsm.states[initial].android_context.activity_name
    siblings = (
        sorted(
            state_id
            for state_id, state in fsm.states.items()
            if state_id != initial
            and state.android_context.activity_name == activity
            and state.hierarchy_level is not HierarchyLevel.COMPONENT
        )
        if activity
        else []
    )
    return [initial, *siblings][:_MAX_ENTRY_CANDIDATES]


def _entry_candidate(fsm: AppFSM, state_id: str) -> dict[str, Any]:
    """Describe one candidate entry screen and the edges that already leave it."""
    state = fsm.states[state_id]
    action_surface = [
        {
            "action": _action_summary(transition.action),
            "target_state_id": transition.target,
            "target_label": _state_label(fsm, transition.target),
        }
        for transition in fsm.transitions
        if transition.source == state_id
    ]
    return {
        "state_id": state_id,
        "label": _state_label(fsm, state_id),
        "page_function": state.annotations.page_function,
        "activity_name": state.android_context.activity_name,
        "is_initial_state": state_id == fsm.initial_state,
        "action_surface": action_surface[:_MAX_ENTRY_CANDIDATE_ACTIONS],
        "action_surface_truncated": len(action_surface) > _MAX_ENTRY_CANDIDATE_ACTIONS,
    }


def _cold_start_guidance(fsm: AppFSM, consecutive_unlocalized_rounds: int) -> dict[str, Any] | None:
    """Evidence-only slice for an unlocalized screen with no anchor to fall back on.

    An episode that starts on an unmodeled screen has no last-localized state, so
    the anchored guidance below cannot be built and the agent used to receive one
    bare sentence. This returns the model's own entry screens instead. It is
    evidence about the *model*, never a claim about the current screen: no trust
    flag moves, no action is authorized, and the caller's verdict stays DENY.

    Returns:
        The guidance payload, or ``None`` when the model records no entry screen.
    """
    candidates = _entry_candidate_state_ids(fsm)
    if not candidates:
        return None
    guidance: dict[str, Any] = {
        "status": "unlocalized_no_anchor",
        "statement": _COLD_START_STATEMENT,
        "last_localized_state": None,
        "entry_candidates": [_entry_candidate(fsm, state_id) for state_id in candidates],
        "entry_candidates_caveat": _ENTRY_CANDIDATES_CAVEAT,
        "execution_authorization": False,
        "consecutive_unlocalized_rounds": consecutive_unlocalized_rounds,
    }
    if consecutive_unlocalized_rounds >= _UNLOCALIZED_ROUNDS_ESCALATION_THRESHOLD:
        guidance["escalation"] = _COLD_START_ESCALATION
    return guidance


def build_unlocalized_guidance(
    fsm: AppFSM,
    last_localized_state_id: str | None,
    consecutive_unlocalized_rounds: int = 0,
) -> dict[str, Any] | None:
    """Structural dead-end guidance for a screen Vigil cannot localize.

    With a usable anchor the payload names the block, the last modeled screen,
    and the edges that already exist out of it. Without one — no anchor at all,
    or an anchor this EFSM does not know, since a stale anchor is not evidence —
    it falls back to :func:`_cold_start_guidance`, which lists the model's entry
    screens instead. ``None`` is returned only when the model records no entry
    screen either.

    Nothing in either shape is a verdict, a plan, or a suggested next action.

    Args:
        fsm: The app's verified FSM.
        last_localized_state_id: The most recent state the bridge localized in
            this episode, as reported by the runner.
        consecutive_unlocalized_rounds: Unlocalized planner rounds seen in a row
            before this one, used only to escalate the wording.

    Returns:
        The guidance payload, or ``None`` when neither an anchor nor an entry
        screen is available.
    """
    if last_localized_state_id is None or last_localized_state_id not in fsm.states:
        return _cold_start_guidance(fsm, consecutive_unlocalized_rounds)
    state = fsm.states[last_localized_state_id]
    guidance: dict[str, Any] = {
        "status": "unlocalized",
        "statement": _UNLOCALIZED_STATEMENT,
        "last_localized_state": {
            "state_id": last_localized_state_id,
            "label": _state_label(fsm, last_localized_state_id),
            "page_function": state.annotations.page_function,
        },
        "recovery_actions": _recovery_actions(fsm, last_localized_state_id),
        "recovery_actions_caveat": _RECOVERY_ACTIONS_CAVEAT,
        "consecutive_unlocalized_rounds": consecutive_unlocalized_rounds,
    }
    if consecutive_unlocalized_rounds >= _UNLOCALIZED_ROUNDS_ESCALATION_THRESHOLD:
        guidance["escalation"] = _UNLOCALIZED_ESCALATION
    return guidance


def _unlocalized_guidance_prose(guidance: dict[str, Any] | None) -> str:
    """Render the unlocalized guidance as the planner ``guidance`` string."""
    if guidance is None:
        return "Vigil could not localize this screen."
    sentences = [f"Vigil could not localize this screen. {guidance['statement']}"]
    anchor = guidance["last_localized_state"]
    if anchor is None:
        candidates = guidance["entry_candidates"]
        sentences.append(
            f"No state has been localized yet. The model's {len(candidates)} entry "
            "screen(s) and their existing edges are listed in "
            "unlocalized_guidance.entry_candidates as evidence only; none of them is "
            "a match for this screen."
        )
    else:
        sentences.append(f"Last localized state: {anchor['state_id']} ({anchor['label']}).")
        if guidance["recovery_actions"]:
            sentences.append(
                f"{len(guidance['recovery_actions'])} modeled edge(s) out of that state are "
                "listed in unlocalized_guidance.recovery_actions; they may be stale."
            )
    prose = " ".join(sentences)
    if "escalation" in guidance:
        return f"{guidance['escalation']} {prose}"
    return prose


def build_current_state_guidance(
    fsm: AppFSM,
    state_id: str,
    outgoing: list[Any],
    selectable: list[bool],
    binding_snapshot: RevisionedIntentBindingSnapshot | dict[str, Any],
    live_ui_elements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the current-state EFSM slice injected into M3A planner prompt."""
    transition_guides = [
        _transition_guide(fsm, transition, flag, live_ui_elements=live_ui_elements)
        for transition, flag in zip(outgoing, selectable, strict=True)
    ]
    snapshot = _coerce_binding_snapshot(binding_snapshot)
    guidance = {
        "guide_type": "current_state_efsm_slice",
        "efsm_revision": fsm.efsm_revision,
        "current_state": _state_summary(fsm, state_id),
        "source_state_invariants": _invariant_summaries(fsm, state_id),
        "outgoing_transitions": transition_guides,
        "binding_snapshot": snapshot.model_dump(mode="json"),
    }
    guidance = _with_clause_activation_guidance(guidance, outgoing, snapshot)
    demand = derive_intent_patch_demand(
        state_id=state_id,
        outgoing=outgoing,
        selectable=selectable,
        snapshot=snapshot,
        efsm_revision=fsm.efsm_revision,
    )
    guidance["intent_patch_request"] = demand.request
    return guidance


# --- planner mode --------------------------------------------------------


def build_planner_response(
    fsm: AppFSM,
    screen: RawScreen,
    binding_snapshot: RevisionedIntentBindingSnapshot | dict[str, Any],
    live_ui_elements: list[dict[str, Any]] | None = None,
    last_localized_state_id: str | None = None,
    consecutive_unlocalized_rounds: int = 0,
) -> dict[str, Any]:
    stage_timings_ms: dict[str, float] = {}
    _stage_t0 = time.perf_counter()
    location = StateLocator(fsm).locate(screen)
    stage_timings_ms["localization_ms"] = (time.perf_counter() - _stage_t0) * 1000.0
    known = is_known_location(location.result) and location.state_id is not None
    if not known:
        # An unlocalized screen has no known state or slot schema, so the only admissible
        # patch is the empty (zero-semantic-demand) one. Emitting a real request here lets
        # the M3A boundary complete planner -> empty-patch admission -> admitted proposal
        # -> execution, which is what makes a topology (STATE_UNKNOWN) continuation
        # reachable end to end.
        snapshot = _coerce_binding_snapshot(binding_snapshot)
        _stage_t0 = time.perf_counter()
        unlocalized = build_unlocalized_guidance(
            fsm,
            last_localized_state_id,
            consecutive_unlocalized_rounds,
        )
        stage_timings_ms["guidance_build_ms"] = (time.perf_counter() - _stage_t0) * 1000.0
        return {
            "stage_timings_ms": stage_timings_ms,
            "verdict": "UNCERTAIN",
            "state_id": None,
            "allowed_actions": [],
            "intent_patch_request": _unknown_state_intent_patch_request(fsm, snapshot),
            "unlocalized_guidance": unlocalized,
            "guidance": _unlocalized_guidance_prose(unlocalized),
        }

    state_id = location.state_id
    outgoing = [t for t in fsm.transitions if t.source == state_id]
    descriptors = [_action_descriptor(t.action, t.target, live_ui_elements) for t in outgoing]

    # Annotate each descriptor with the planner-time selectable flag (goal-free).
    _stage_t0 = time.perf_counter()
    selectable, verification = _planner_selectability(fsm, state_id, outgoing)
    stage_timings_ms["action_mask_ms"] = (time.perf_counter() - _stage_t0) * 1000.0
    for descriptor, flag in zip(descriptors, selectable, strict=True):
        descriptor["selectable"] = flag

    _stage_t0 = time.perf_counter()
    current_guidance = build_current_state_guidance(
        fsm,
        state_id,
        outgoing,
        selectable,
        binding_snapshot=binding_snapshot,
        live_ui_elements=live_ui_elements,
    )
    # Derived readiness: a structurally selectable action can still be blocked at
    # execution time by an unbound intent slot. Reuse the guidance activation rather
    # than re-deriving it so descriptor, guidance, and runtime stay consistent.
    readiness = [
        _transition_readiness(transition, guide, flag)
        for transition, guide, flag in zip(
            outgoing,
            current_guidance["outgoing_transitions"],
            selectable,
            strict=True,
        )
    ]
    for descriptor, (ready, _note) in zip(descriptors, readiness, strict=True):
        descriptor["ready"] = ready
    stage_timings_ms["guidance_build_ms"] = (time.perf_counter() - _stage_t0) * 1000.0

    # The planner response carries the patch request once at top level; drop the
    # nested copy so the same request is never serialized twice into one prompt.
    intent_patch_request = current_guidance.pop("intent_patch_request")
    _stage_t0 = time.perf_counter()
    planner_feedback = _build_planner_feedback(
        fsm,
        screen,
        state_id,
        outgoing,
        selectable,
        current_guidance,
    )
    stage_timings_ms["feedback_preview_ms"] = (time.perf_counter() - _stage_t0) * 1000.0

    if not descriptors:
        return {
            "stage_timings_ms": stage_timings_ms,
            "verdict": "UNCERTAIN",
            "state_id": state_id,
            "allowed_actions": [],
            "agent_efsm_guide": build_agent_efsm_guide(fsm),
            "current_state_guidance": current_guidance,
            "intent_patch_request": intent_patch_request,
            "vigil_feedback": planner_feedback,
            "guidance": f"Vigil localized state {state_id} but found no legal outgoing action.",
        }

    allowed = [descriptor for descriptor in descriptors if descriptor["selectable"]]
    unavailable = [descriptor for descriptor in descriptors if not descriptor["selectable"]]
    if not allowed:
        return {
            "stage_timings_ms": stage_timings_ms,
            "verdict": _verdict(verification.overall_result),
            "state_id": state_id,
            "allowed_actions": [],
            "unavailable_actions": unavailable,
            "agent_efsm_guide": build_agent_efsm_guide(fsm),
            "current_state_guidance": current_guidance,
            "intent_patch_request": intent_patch_request,
            "vigil_feedback": planner_feedback,
            "guidance": (
                f"Vigil localized state {state_id}; outgoing topology is shown as "
                "evidence, but no candidate currently satisfies planner-time trust gates."
            ),
        }

    allowed_labels = [
        f"{_descriptor_label(descriptor)} {note}" if note else _descriptor_label(descriptor)
        for descriptor, (_ready, note) in zip(descriptors, readiness, strict=True)
        if descriptor["selectable"]
    ]
    guidance = (
        f"Vigil: prefer these legal actions at state {state_id}; they satisfy "
        "planner-time trust gates: " + ", ".join(allowed_labels) + "."
    )
    return {
        "stage_timings_ms": stage_timings_ms,
        "verdict": "ALLOW",
        "state_id": state_id,
        "allowed_actions": allowed,
        "unavailable_actions": unavailable,
        "agent_efsm_guide": build_agent_efsm_guide(fsm),
        "current_state_guidance": current_guidance,
        "intent_patch_request": intent_patch_request,
        "vigil_feedback": planner_feedback,
        "guidance": guidance,
    }


def _unknown_state_intent_patch_request(
    fsm: AppFSM, snapshot: RevisionedIntentBindingSnapshot
) -> dict[str, Any]:
    """The canonical zero-semantic-demand patch request for an unlocalized screen.

    No state or slot schema is known, so no slot may be demanded; only an empty patch is
    admissible. The request still pins the binding base revision and the EFSM revision so
    admission can reject a stale or forged unknown-state request.
    """
    revision = fsm.efsm_revision
    return {
        "request_id": f"ipr:__unknown__:m{revision}:r{snapshot.revision}:-",
        "state_id": None,
        "efsm_revision": revision,
        "base_revision": snapshot.revision,
        "requested_slots": [],
        "unknown_state": True,
    }


def _admit_unknown_state_patch(
    fsm: AppFSM,
    intent_patch_request: dict[str, Any],
    intent_patch: IntentPatch | dict[str, Any],
    snapshot: RevisionedIntentBindingSnapshot,
    proposed_action: dict[str, Any],
    observe_bailed: bool = False,
) -> dict[str, Any]:
    """Admit the empty patch for an unlocalized screen into a real AdmittedPlannerProposal.

    Validates the request against the canonical zero-demand form (which pins base and EFSM
    revision) and requires an empty patch; a non-empty update is rejected. The admitted
    proposal carries the unchanged binding snapshot (an empty patch never bumps revision).

    ``observe_bailed``: after an observe-policy bail the episode has already failed
    for Vigil and the agent proceeds on its own; a rejected patch would only stall
    the round (the agent retries the same proposal against a gate that cannot open).
    The only admissible unknown-state patch is the empty one, so post-bail a
    non-empty patch is admitted WITH ITS UPDATES DROPPED -- no value ever enters
    the frozen snapshot, the drop is recorded, and enforcement semantics before
    the bail are untouched.
    """
    reasons: list[str] = []
    dropped_updates: dict[str, Any] = {}
    expected = _unknown_state_intent_patch_request(fsm, snapshot)
    if intent_patch_request != expected:
        reasons.append("intent_patch_request_stale")
    try:
        patch = IntentPatch.model_validate(intent_patch)
    except ValidationError:
        patch = None
        reasons.append("invalid_intent_patch_shape")
    if patch is not None:
        if patch.base_revision != snapshot.revision:
            reasons.append(f"base_revision_mismatch:{patch.base_revision}!={snapshot.revision}")
        if patch.updates and observe_bailed and not reasons:
            dropped_updates = dict(patch.updates)
            patch = IntentPatch(base_revision=patch.base_revision, updates={})
        elif patch.updates:
            reasons.append("unknown_state_requires_empty_patch")

    request_id = expected["request_id"]
    if reasons:
        return {
            "type": "vigil_intent_patch_admission",
            "admitted": False,
            "state_id": None,
            "request_id": request_id,
            "reasons": reasons,
            "current_state_guidance": None,
        }

    canonical_action = canonical_agent_action_proposal(_opaque_m3a_action_proposal(proposed_action))
    proposal = AdmittedPlannerProposal(
        proposal_id=(
            f"proposal:{request_id}:m{fsm.efsm_revision}:r{snapshot.revision}:{canonical_action}"
        ),
        request_id=request_id,
        efsm_revision=fsm.efsm_revision,
        agent_action_proposal=canonical_action,
        binding_snapshot=snapshot,
    )
    return {
        "type": "vigil_intent_patch_admission",
        "admitted": True,
        "state_id": None,
        "request_id": request_id,
        "reasons": [],
        "current_state_guidance": None,
        "admitted_proposal": proposal.model_dump(mode="json"),
        # Post-bail drop-and-admit record: which updates were withheld from the
        # snapshot so the round could proceed. Empty outside observe post-bail.
        "observe_dropped_updates": dropped_updates,
    }


#: Rejection-reason prefixes the observe post-bail drop-and-admit may discard.
#: Only value-grounding refusals qualify: the value is real agent intent that the
#: goal/screen cannot attest, so dropping it (slot stays missing) is exactly the
#: conservative outcome admission would produce for an omitted slot. Structural
#: rejections (stale request, shape, revision) are never dropped around.
_OBSERVE_DROPPABLE_REASON_PREFIXES = ("ungrounded_string_value:", "ungrounded_numeric_value:")


def build_intent_patch_response(
    fsm: AppFSM,
    intent_patch_request: dict[str, Any],
    intent_patch: IntentPatch | dict[str, Any],
    binding_snapshot: RevisionedIntentBindingSnapshot | dict[str, Any],
    proposed_action: dict[str, Any],
    goal_text: str | None = None,
    screen_texts: Sequence[str] = (),
    observe_bailed: bool = False,
) -> dict[str, Any]:
    """Admit the patch emitted beside one M3A planner action proposal.

    The caller-provided request is correlation data only. Vigil re-derives the
    current state's selectable transition demand from the FSM before admitting
    any update. An unlocalized (unknown-state) request routes to the zero-demand
    admission path, which never weakens known-state admission. ``goal_text`` (the
    episode's public goal, supplied by the runner) additionally gates patch
    values on goal derivability; ``None`` preserves schema-only admission.
    """
    snapshot = _coerce_binding_snapshot(binding_snapshot)
    if (
        intent_patch_request.get("unknown_state") is True
        or intent_patch_request.get("state_id") is None
    ):
        return _admit_unknown_state_patch(
            fsm,
            intent_patch_request,
            intent_patch,
            snapshot,
            proposed_action,
            observe_bailed=observe_bailed,
        )
    state_id = intent_patch_request.get("state_id")
    if not isinstance(state_id, str) or state_id not in fsm.states:
        raise ValueError("intent_patch_request must identify an FSM state")

    outgoing = [transition for transition in fsm.transitions if transition.source == state_id]
    selectable, _ = _planner_selectability(fsm, state_id, outgoing)
    demand = derive_intent_patch_demand(
        state_id=state_id,
        outgoing=outgoing,
        selectable=selectable,
        snapshot=snapshot,
        efsm_revision=fsm.efsm_revision,
    )

    reasons: tuple[str, ...] = ()
    if intent_patch_request != demand.request:
        reasons = ("intent_patch_request_stale",)
    try:
        patch = IntentPatch.model_validate(intent_patch)
    except ValidationError:
        patch = None
        reasons = (*reasons, "invalid_intent_patch_shape")

    if reasons or patch is None:
        current_guidance = build_current_state_guidance(
            fsm,
            state_id,
            outgoing,
            selectable,
            binding_snapshot=snapshot,
        )
        return {
            "type": "vigil_intent_patch_admission",
            "admitted": False,
            "state_id": state_id,
            "request_id": demand.request["request_id"],
            "reasons": list(reasons),
            "current_state_guidance": current_guidance,
        }

    result = admit_intent_patch(
        base_snapshot=snapshot,
        patch=patch,
        demand=demand,
        agent_action_proposal=_opaque_m3a_action_proposal(proposed_action),
        goal_text=goal_text,
        screen_texts=screen_texts,
    )
    observe_dropped_updates: dict[str, Any] = {}
    dropped_update_reasons: list[str] = []
    if not result.admitted and observe_bailed and result.reasons:
        # Post-bail drop-and-admit: the episode already failed for Vigil and the
        # agent proceeds on its own; a value-grounding rejection would only stall
        # the round against a gate that cannot open. Drop exactly the refused
        # slots (they stay missing in the snapshot -- the same outcome as the
        # agent omitting them) and re-admit; any structural rejection reason
        # keeps the full rejection. Pre-bail behavior is byte-identical.
        droppable = {
            reason.split(":", 1)[1]
            for reason in result.reasons
            if reason.startswith(_OBSERVE_DROPPABLE_REASON_PREFIXES)
        }
        all_droppable = droppable and all(
            reason.startswith(_OBSERVE_DROPPABLE_REASON_PREFIXES) for reason in result.reasons
        )
        if all_droppable:
            reduced = IntentPatch(
                base_revision=patch.base_revision,
                updates={
                    name: value for name, value in patch.updates.items() if name not in droppable
                },
            )
            retried = admit_intent_patch(
                base_snapshot=snapshot,
                patch=reduced,
                demand=demand,
                agent_action_proposal=_opaque_m3a_action_proposal(proposed_action),
                goal_text=goal_text,
                screen_texts=screen_texts,
            )
            if retried.admitted:
                observe_dropped_updates = {
                    name: patch.updates.get(name) for name in sorted(droppable)
                }
                dropped_update_reasons = list(result.reasons)
                result = retried
    result_snapshot = (
        result.admitted_proposal.binding_snapshot
        if result.admitted_proposal is not None
        else snapshot
    )
    current_guidance = build_current_state_guidance(
        fsm,
        state_id,
        outgoing,
        selectable,
        binding_snapshot=result_snapshot,
    )
    response = {
        "type": "vigil_intent_patch_admission",
        "admitted": result.admitted,
        "state_id": state_id,
        "request_id": demand.request["request_id"],
        "reasons": list(result.reasons),
        "current_state_guidance": current_guidance,
        # Post-bail drop-and-admit record (empty outside observe post-bail):
        # which updates were withheld, and the grounding reasons that refused them.
        "observe_dropped_updates": observe_dropped_updates,
        "dropped_update_reasons": dropped_update_reasons,
    }
    if result.admitted_proposal is not None:
        response["admitted_proposal"] = result.admitted_proposal.model_dump(mode="json")
    return response


def build_intent_rebind_response(
    fsm: AppFSM,
    binding_snapshot: RevisionedIntentBindingSnapshot | dict[str, Any],
    intent_patch: IntentPatch | dict[str, Any],
    recovery_authorization: RecoveryAuthorization | dict[str, Any],
    proposed_action: dict[str, Any],
) -> dict[str, Any]:
    """Admit one evidence-triggered rebind against a Vigil-issued authorization.

    Overwrites only the authorization's eligible slots and freezes a new revision
    on a value change (a same-value proposal is a no-op). This never returns a
    verdict: the revised proposal must pass back through the full verifier. The
    per-chain attempt budget is owned by the caller's recovery ledger.
    """
    snapshot = _coerce_binding_snapshot(binding_snapshot)
    authorization = RecoveryAuthorization.model_validate(recovery_authorization)
    try:
        patch = IntentPatch.model_validate(intent_patch)
    except ValidationError:
        return {
            "type": "vigil_intent_rebind_admission",
            "admitted": False,
            "no_op": False,
            "recovery_chain_id": authorization.recovery_chain_id,
            "reasons": ["invalid_intent_patch_shape"],
        }

    result = admit_intent_rebind(
        fsm=fsm,
        base_snapshot=snapshot,
        patch=patch,
        authorization=authorization,
        agent_action_proposal=_opaque_m3a_action_proposal(proposed_action),
    )
    response: dict[str, Any] = {
        "type": "vigil_intent_rebind_admission",
        "admitted": result.admitted,
        "no_op": result.no_op,
        "recovery_chain_id": authorization.recovery_chain_id,
        "reasons": list(result.reasons),
    }
    if result.admitted_proposal is not None:
        response["admitted_proposal"] = result.admitted_proposal.model_dump(mode="json")
    return response


def _build_planner_feedback(
    fsm: AppFSM,
    screen: RawScreen,
    state_id: str,
    outgoing: list[Any],
    selectable: list[bool],
    current_guidance: dict[str, Any],
) -> dict[str, Any] | None:
    """Evidence-only preview of admitted guard clauses and their activation state.

    This is a PREVIEW: no execution verifier pass exists at planner time, so each
    predicate item is tagged ``preview`` and must never be read as a real
    execution witness. It evaluates admitted clauses against the current screen
    only to illustrate activation, using the same evaluator (never a re-parse).
    """
    snapshot = RevisionedIntentBindingSnapshot.model_validate(current_guidance["binding_snapshot"])
    evaluator = DSLEvaluator()
    activator = ClauseActivator()
    screen_ctx = DecisionEngine._build_screen_context(screen)
    best_feedback: dict[str, Any] | None = None
    best_score = 0

    for transition, flag in zip(outgoing, selectable, strict=True):
        contract = transition.guard_contract
        if (
            not flag
            or contract is None
            or contract.admission_status is not GuardAdmissionStatus.ADMITTED
            or not contract.clauses
        ):
            continue
        activation = activator.activate(contract, snapshot.values)
        action_ctx = DecisionEngine._build_minimal_action_context(
            canonical_transition_action(transition.action)
        )
        evaluation_intent = IntentContext(variables=activation.bound_variables)
        preview = (
            evaluator.evaluate(
                activation.expression,
                intent_ctx=evaluation_intent,
                screen_ctx=screen_ctx,
                action_ctx=action_ctx,
                record_witness=True,
            )
            if activation.expression is not None
            else None
        )
        predicate_checks = _witness_predicate_checks(
            preview.witness if preview is not None else None,
            activation,
            preview=True,
        )

        evidence_transition = {
            "source": transition.source,
            "target": transition.target,
            "action": _action_summary(transition.action),
        }
        evidence_transition["action"].setdefault("action_type", transition.action.get("type"))
        feedback = {
            "type": "vigil_verification_observation",
            "phase": "planner",
            "intervention_point": "pre_action_planning",
            "event_id": _event_id(
                state_id,
                "planner_intent_evidence",
                "OBSERVE",
                evidence_transition["action"],
            ),
            "verdict": "OBSERVE",
            "reason": "planner_intent_evidence",
            "current_state": {
                "status": "localized",
                "state_id": state_id,
                "label": _state_label(fsm, state_id) or None,
            },
            "evidence_transition": evidence_transition,
            "verification_evidence": {
                "checks": [
                    _guard_check(
                        transition,
                        "unknown",
                        (
                            "Planner-time guard predicates expose intent binding "
                            "evidence before execution."
                        ),
                    ),
                    *(
                        {
                            "kind": "clause_activation",
                            **clause.model_dump(mode="json"),
                        }
                        for clause in activation.clauses
                    ),
                    *predicate_checks,
                ]
            },
            "process_constraints": list(_PLANNER_CONSTRAINTS),
            "binding_snapshot": current_guidance["binding_snapshot"],
            "intent_contract": {"slots": _transition_slots(transition)},
        }
        score = len(activation.clauses)
        if score > best_score:
            best_score = score
            best_feedback = feedback
    return best_feedback


# --- execution mode ------------------------------------------------------


# Evidence-only feedback: mapping the runtime verdict/reason into an
# app-agnostic verifier observation. This never contains repair options, a
# suggested next action, hidden intent, gold verdicts, or benchmark oracles.


def _guard_checks(
    fsm: AppFSM,
    result: RuntimeVerificationResult,
    reason: str,
) -> tuple[list[dict[str, Any]], Any]:
    """Build guard evidence checks; also return the resolved transition.

    Predicate evidence comes from the real evaluator pass captured on
    ``result.final_result.predicate_witness`` — the adapter never re-parses or
    re-evaluates the guard. Returns ``(checks, transition)``; ``transition`` is
    ``None`` when it cannot be resolved (e.g. no localized pre-state).
    """
    pre_state_id = result.pre_state_id
    transition = None
    if pre_state_id is not None:
        transition = fsm.resolve_transition(pre_state_id, result.bound_action).transition
    guard_status = "false" if reason == "guard_failed" else "unknown"
    checks: list[dict[str, Any]] = [
        _guard_check(transition, guard_status, result.final_result.details)
    ]
    if transition is not None:
        activation = result.final_result.clause_activation
        if activation is not None:
            for clause in activation.clauses:
                checks.append(
                    {
                        "kind": "clause_activation",
                        **clause.model_dump(mode="json"),
                    }
                )
        checks.extend(_witness_predicate_checks(result.final_result.predicate_witness, activation))
    return checks, transition


def _build_execution_feedback(
    fsm: AppFSM,
    screen: RawScreen,
    result: RuntimeVerificationResult,
    binding_snapshot: RevisionedIntentBindingSnapshot,
    last_localized_state_id: str | None = None,
) -> dict[str, Any]:
    """Evidence-only verifier observation for a refused (DENY) result."""
    final = result.final_result
    reason = final.reason.value
    pre_state_id = result.pre_state_id

    checks: list[dict[str, Any]] = []
    transition = None

    if reason in (
        "guard_failed",
        "guard_inconclusive",
        "semantic_guard_coverage_gap",
    ):
        checks, transition = _guard_checks(fsm, result, reason)
    elif reason == "action_ambiguous":
        checks.append(
            {
                "kind": "action_binding",
                "binding_evidence": {
                    "binding_status": result.binding_status.value,
                    "binding_reason": result.binding_reason,
                },
            }
        )
    elif reason in (
        "transition_not_in_fsm",
        "state_unknown",
        "state_similar_fuzzy_match",
        "incomplete_coverage",
    ):
        outgoing = (
            [t for t in fsm.transitions if t.source == pre_state_id]
            if pre_state_id is not None
            else []
        )
        checks.append(
            {
                "kind": "transition",
                "transition_evidence": {
                    "source_state_id": pre_state_id,
                    "resolved_transition": None,
                    "available_action_summaries": [_action_summary(t.action) for t in outgoing],
                },
            }
        )
    elif reason in ("invariant_failed", "invariant_inconclusive"):
        checks.append(
            {
                "kind": "invariant",
                "details": final.details,
                "target_invariants": _invariant_summaries(fsm, final.target_state_id),
            }
        )

    checked_action = _action_summary(result.bound_action, include_value_payload=True)
    checked_action.setdefault("action_type", result.bound_action.get("type"))

    feedback: dict[str, Any] = {
        "type": "vigil_verification_observation",
        "phase": "execution",
        "intervention_point": "pre_action",
        "event_id": _event_id(pre_state_id, reason, _verdict(final.result), checked_action),
        "action_executed": False,
        "verdict": _verdict(final.result),
        "proven_refutation": final.proven_refutation,
        "reason": reason,
        "current_state": {
            "status": "localized" if pre_state_id is not None else "unknown",
            "state_id": pre_state_id,
            "label": _state_label(fsm, pre_state_id) or None,
        },
        "checked_action": checked_action,
        "verification_evidence": {"checks": checks},
        "process_constraints": list(_PROCESS_CONSTRAINTS),
    }

    # A screen the verifier could not localize is a dead end for this proposal: no
    # transition resolves and no intent patch can release it. Carry the same structural
    # dead-end guidance the planner emits so the blocked observation names a way out.
    if reason == "state_unknown":
        unlocalized = build_unlocalized_guidance(fsm, last_localized_state_id)
        if unlocalized is not None:
            feedback["unlocalized_guidance"] = unlocalized

    # A proven guard failure additionally surfaces the refuting operands at top level:
    # the same FALSE witnesses already nested in verification_evidence, flattened so the
    # agent does not have to walk the evidence tree to see the concrete conflict.
    if reason == "guard_failed":
        refuted = _refuted_predicates(
            final.predicate_witness,
            final.clause_activation,
        )
        if refuted:
            feedback["refuted_predicates"] = refuted

    # Attach only the admitted typed intent contract and the exact agent snapshot.
    if transition is not None and _transition_slots(transition):
        feedback["binding_snapshot"] = binding_snapshot.model_dump(mode="json")
        feedback["intent_contract"] = {"slots": _transition_slots(transition)}

    return feedback


def _build_unsupported_feedback(m3a_action: dict[str, Any]) -> dict[str, Any]:
    """Evidence-only observation for an M3A action with no Vigil executable form."""
    action_type = m3a_action.get("action_type")
    checked_action = {"action_type": action_type}
    details = (
        f"M3A action_type {action_type!r} has no Vigil executable form; blocking at execution time."
    )
    return {
        "type": "vigil_verification_observation",
        "phase": "execution",
        "intervention_point": "pre_action",
        "event_id": _event_id(None, "unsupported_action", "UNCERTAIN", checked_action),
        "action_executed": False,
        "verdict": "UNCERTAIN",
        "reason": "unsupported_action",
        "current_state": {"status": "unknown", "state_id": None, "label": None},
        "checked_action": checked_action,
        "verification_evidence": {"checks": [{"kind": "unsupported_action", "details": details}]},
        "process_constraints": list(_PROCESS_CONSTRAINTS),
    }


def _event_id(
    pre_state_id: str | None, reason: str, verdict: str, checked_action: dict[str, Any]
) -> str:
    """Deterministic observation id (stable across identical inputs; no clock/RNG)."""
    payload = json.dumps(
        {
            "pre_state_id": pre_state_id,
            "reason": reason,
            "verdict": verdict,
            "checked_action": checked_action,
        },
        sort_keys=True,
    )
    return "evt_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _blocked_execution_response(
    proposal: AdmittedPlannerProposal,
    reason: str,
    details: str,
    vigil_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Uniform blocking response for pre-verification protocol failures.

    A protocol failure is a refusal for lack of proof: the bridge could not even
    reach the verifier, so it certainly cannot authorize. These reasons are not
    VerifyReason members, but they sit on the conservative (UNCERTAIN) side of the
    same line -- none of them claims the action is unsafe, so ``proven_refutation``
    is explicitly False.
    """
    return {
        "verdict": "UNCERTAIN",
        "proven_refutation": False,
        "reason": reason,
        "details": details,
        "proposal_id": proposal.proposal_id,
        "efsm_revision": proposal.efsm_revision,
        "pre_state_id": None,
        "expected_target_state_id": None,
        "binding_status": None,
        "binding_snapshot": proposal.binding_snapshot.model_dump(mode="json"),
        "clause_activation": None,
        "bound_action": None,
        "should_execute": False,
        "execution_disposition": "BLOCK",
        "continuation_authorized": False,
        "semantic_evolution_requested": False,
        "continuation_receipt": None,
        "semantic_guard_gap": None,
        "vigil_feedback": vigil_feedback,
    }


def _open_app_refusal_feedback(app_name: object, reason: str, details: str) -> dict[str, Any]:
    """Evidence-only observation for a refused ``open_app`` at execution time.

    Carries the refusal-specific ``reason`` so an unresolved display name can never be
    read as the shield withholding a verified-unsafe action: ``open_app`` is refused
    here because Vigil could not establish *which* application the free-text name
    launches, not because a transition, guard, or invariant was refuted.
    """
    checked_action = {"action_type": "open_app", "app_name": app_name}
    return {
        "type": "vigil_verification_observation",
        "phase": "execution",
        "intervention_point": "pre_action",
        "event_id": _event_id(None, reason, "UNCERTAIN", checked_action),
        "action_executed": False,
        "verdict": "UNCERTAIN",
        "reason": reason,
        "current_state": {"status": "unknown", "state_id": None, "label": None},
        "checked_action": checked_action,
        "verification_evidence": {"checks": [{"kind": reason, "details": details}]},
        "process_constraints": list(_PROCESS_CONSTRAINTS),
    }


def _out_of_model_entry_response(
    proposal: AdmittedPlannerProposal, app_name: str, app_package: str
) -> dict[str, Any]:
    """Permit launching the guarded app without adjudicating it as a GUI transition.

    Every AndroidWorld task starts on the home screen, so ``open_app`` is the entry
    point into the very application Vigil exists to guard.  It is not a meta-action like
    ``status``/``answer``/``wait``, but it is also not a member of Sigma: no FSM contains
    an ``open_app`` edge, and the launcher screen is outside the model entirely.

    So this response deliberately makes *no* claim, and says so by carrying no
    verdict at all.  ``verdict`` is ``None``: this action was NOT VERIFIED, which is a
    different thing from being refused.  Reporting ``DENY`` or ``UNCERTAIN`` here would
    be incoherent with ``should_execute`` ``True`` -- "refused, but execute it" is not a
    position the shield can hold, and even UNCERTAIN claims the verifier ran and could
    not decide, which it did not -- while ``ALLOW`` would claim a verification that
    never happened.  ``verification_scope`` names the out-of-scope case explicitly so
    no consumer has to infer it from a null.

    It never reports ``ALLOW``, so ``verified_execution_count`` -- which requires
    ``verdict == "ALLOW"`` -- cannot be inflated by an episode's opening step.  It
    carries no localization, no bound action, no transition, no continuation receipt and
    no semantic-gap evidence, so it cannot reach
    ``ActionBinder``/``StateLocator``/``FsmChecker``, cannot be counted as an unverified
    exposure, and cannot let runtime evolution fabricate an entry-point edge in
    the transition graph. The action alphabet, transition graph, ``ActionType``,
    and ``canonical_action_key`` are all untouched.
    """
    return {
        "verdict": None,
        "verification_scope": "out_of_model",
        "reason": "out_of_model_entry",
        "details": (
            f"open_app({app_name!r}) targets the guarded package {app_package!r}. "
            "Launching the guarded app is outside the model's action alphabet; Vigil "
            "adjudicates screens of this app, not the launcher. Permitted without "
            "verification and without recording a transition."
        ),
        "proposal_id": proposal.proposal_id,
        "efsm_revision": proposal.efsm_revision,
        "pre_state_id": None,
        "expected_target_state_id": None,
        "binding_status": None,
        "binding_snapshot": proposal.binding_snapshot.model_dump(mode="json"),
        "clause_activation": None,
        "bound_action": None,
        "should_execute": True,
        "execution_disposition": "OUT_OF_MODEL_ENTRY",
        "continuation_authorized": False,
        "semantic_evolution_requested": False,
        "continuation_receipt": None,
        "semantic_guard_gap": None,
        "vigil_feedback": {
            "type": "vigil_verification_observation",
            "phase": "execution",
            "intervention_point": "pre_action",
            "verdict": None,
            "verification_scope": "out_of_model",
            "reason": "out_of_model_entry",
            "action_executed": True,
            "checked_action": {"action_type": "open_app", "app_name": app_name},
            "current_state": {"state_id": None, "label": None, "status": "out_of_scope"},
            "process_constraints": [
                "Vigil has no model of the launcher and made no claim about this action.",
                "Verification resumes once the screen belongs to the guarded app.",
            ],
            "verification_evidence": {
                "checks": [
                    {
                        "kind": "out_of_model_entry",
                        "details": f"launch target resolves to guarded package {app_package!r}",
                    }
                ]
            },
        },
    }


def _out_of_model_view_response(
    proposal: AdmittedPlannerProposal,
    m3a_action: dict[str, Any],
    pre_state_id: str | None,
    binding_details: str,
) -> dict[str, Any]:
    """Permit an unverifiable scroll without adjudicating it as a transition.

    A screen-level scroll carries no element target, so it can never bind to a
    stable identity or resolve to an edge in the transition graph.
    Like ``_out_of_model_entry_response`` this makes *no* claim, and says
    so by carrying no verdict: ``verdict`` is ``None`` with
    ``verification_scope`` ``"out_of_model"``, because no verdict value means
    "not adjudicated" -- even UNCERTAIN claims the verifier ran and could not decide.
    DENY or UNCERTAIN would contradict ``should_execute``
    ``True``; ALLOW would claim a verification that never ran
    (``verified_execution_count`` still cannot inflate). No transition is recorded,
    evolution sees no evidence, and the next step's localization re-syncs whatever
    the scroll revealed. Clicks,
    long-presses, and text input never take this path — a mutating action that
    cannot be bound stays withheld.
    """
    checked_action = {
        "action_type": "scroll",
        "direction": m3a_action.get("direction"),
    }
    return {
        "verdict": None,
        "verification_scope": "out_of_model",
        "reason": "out_of_model_view_action",
        "details": (
            "Screen-level scroll has no bindable element target "
            f"({binding_details}); it is a non-committal view gesture outside the "
            "model's action alphabet. Permitted without verification and without "
            "recording a transition; localization re-syncs on the next step."
        ),
        "proposal_id": proposal.proposal_id,
        "efsm_revision": proposal.efsm_revision,
        "pre_state_id": pre_state_id,
        "expected_target_state_id": None,
        "binding_status": None,
        "binding_snapshot": proposal.binding_snapshot.model_dump(mode="json"),
        "clause_activation": None,
        "bound_action": None,
        "should_execute": True,
        "execution_disposition": "OUT_OF_MODEL_VIEW_ACTION",
        "continuation_authorized": False,
        "semantic_evolution_requested": False,
        "continuation_receipt": None,
        "semantic_guard_gap": None,
        "vigil_feedback": {
            "type": "vigil_verification_observation",
            "phase": "execution",
            "intervention_point": "pre_action",
            "event_id": _event_id(
                pre_state_id, "out_of_model_view_action", "NOT_VERIFIED", checked_action
            ),
            "verdict": None,
            "verification_scope": "out_of_model",
            "reason": "out_of_model_view_action",
            "action_executed": True,
            "checked_action": checked_action,
            "current_state_id": pre_state_id,
            "process_constraints": [
                "The scroll executed without verification: it binds to no element and "
                "matches no modeled transition, and it commits nothing.",
                "Verification resumes at the next proposed action; do not treat this "
                "as approval of any subsequent click or input.",
            ],
            "verification_evidence": {
                "checks": [
                    {
                        "kind": "out_of_model_view_action",
                        "details": binding_details,
                    }
                ]
            },
        },
    }


def build_execution_response(
    fsm: AppFSM,
    screen: RawScreen,
    m3a_action: dict[str, Any],
    admitted_proposal: AdmittedPlannerProposal | dict[str, Any],
    semantic_gap_policy: str = "block",
    foreground_package: str | None = None,
    foreground_activity: str | None = None,
    last_localized_state_id: str | None = None,
    open_app_resolved_package: str | None = None,
) -> dict[str, Any]:
    """Verify one execution-time proposal against the loaded EFSM.

    ``open_app_resolved_package`` is the package that the agent's launcher
    would start for the proposal's free-text ``app_name`` — resolved by the *runner*
    through the harness's authoritative name→package mapping (for AndroidWorld,
    ``adb_utils.get_adb_activity``, the same function its executor uses to perform
    ``open_app``). Vigil never guesses the mapping itself: with no resolution supplied
    the launch is refused with the distinct reason ``open_app_unresolved_app_name``.
    """
    if semantic_gap_policy not in _SEMANTIC_GAP_POLICIES:
        raise ValueError(f"semantic_gap_policy must be one of {_SEMANTIC_GAP_POLICIES}")
    proposal = _coerce_admitted_proposal(admitted_proposal)
    if proposal.efsm_revision != fsm.efsm_revision:
        return _blocked_execution_response(
            proposal,
            "planner_efsm_revision_stale",
            (
                f"Planner proposal uses EFSM revision {proposal.efsm_revision}; "
                f"current revision is {fsm.efsm_revision}."
            ),
        )
    canonical_action = canonical_agent_action_proposal(_opaque_m3a_action_proposal(m3a_action))
    if canonical_action != proposal.agent_action_proposal:
        return _blocked_execution_response(
            proposal,
            "planner_proposal_mismatch",
            "Execution action does not match the admitted planner proposal.",
        )
    frozen_bindings = proposal.binding_snapshot
    # Entering the guarded app is the one "unsupported" action that must not deadlock.
    # Checked before RuntimeMonitor so the launch is never adjudicated as a transition.
    # The name→package resolution is the caller's, never a heuristic: Android packages
    # concatenate what display names separate ("simple calendar pro" vs
    # com.simplemobiletools.calendar.pro), so any string matching silently refuses real
    # launches — and a silent fail-closed here is indistinguishable from the all-refused
    # deadlock it was meant to fix. The two refusal reasons below are deliberately
    # distinct from every verifier reason so log readers can never mistake a resolution
    # gap for the shield withholding an unsafe action.
    if m3a_action.get("action_type") == "open_app":
        app_name = m3a_action.get("app_name")
        if open_app_resolved_package and open_app_resolved_package == fsm.app_package:
            return _out_of_model_entry_response(proposal, str(app_name), fsm.app_package)
        if not open_app_resolved_package:
            details = (
                f"open_app({app_name!r}) refused: the display name did not resolve to "
                "any package (no authoritative name→package resolution was supplied "
                "by the runner). This is a resolution failure, not a verifier block."
            )
            print(
                f"[vigil] OPEN_APP_UNRESOLVED app_name={app_name!r}: refused because "
                "the name did not resolve to a package — not an over-blocking verdict",
                file=sys.stderr,
            )
            return _blocked_execution_response(
                proposal,
                "open_app_unresolved_app_name",
                details,
                vigil_feedback=_open_app_refusal_feedback(
                    app_name, "open_app_unresolved_app_name", details
                ),
            )
        details = (
            f"open_app({app_name!r}) resolves to foreign package "
            f"{open_app_resolved_package!r}; the guarded app is {fsm.app_package!r}. "
            "Vigil adjudicates only the application it has a model of."
        )
        return _blocked_execution_response(
            proposal,
            "open_app_foreign_app",
            details,
            vigil_feedback=_open_app_refusal_feedback(app_name, "open_app_foreign_app", details),
        )
    stage_timings_ms: dict[str, float] = {}
    _stage_t0 = time.perf_counter()
    proposed = convert_m3a_action(m3a_action)
    stage_timings_ms["action_canonicalization_ms"] = (time.perf_counter() - _stage_t0) * 1000.0
    if proposed is None:
        return _blocked_execution_response(
            proposal,
            "unsupported_action",
            (
                f"M3A action_type {m3a_action.get('action_type')!r} has no Vigil "
                "executable form; blocking at execution time."
            ),
            vigil_feedback=_build_unsupported_feedback(m3a_action),
        )

    active_monitor = RuntimeMonitor(fsm)
    started = time.perf_counter()
    result = active_monitor.verify_action(
        screen,
        proposed,
        binding_snapshot=frozen_bindings.values,
        goal_state=None,
        stage_timings_out=stage_timings_ms,
    )
    verification_latency_ms = (time.perf_counter() - started) * 1000.0
    stage_timings_ms["verify_total_ms"] = verification_latency_ms
    final = result.final_result
    # A scroll that failed to BIND (no element target on a view gesture) is a
    # non-committal out-of-model action, not a withholding: see
    # _out_of_model_view_response. Only the binding-failure route takes this exit;
    # a scroll that bound but hit a guard/structural verdict is adjudicated
    # normally, and mutating action types never take it.
    if (
        m3a_action.get("action_type") == "scroll"
        and is_conservative_refusal(final)
        and final.reason is VerifyReason.ACTION_AMBIGUOUS
        and result.binding_status is not BindingStatus.MATCH
    ):
        response = _out_of_model_view_response(
            proposal, m3a_action, result.pre_state_id, result.binding_reason
        )
        response["verification_latency_ms"] = verification_latency_ms
        response["stage_timings_ms"] = stage_timings_ms
        return response
    should_execute = final.result is VerifyResult.ALLOW
    _stage_t0 = time.perf_counter()
    continuation = SemanticGapContinuationClassifier(fsm).classify(
        final,
        screen,
        proposed,
    )
    continuation_authorized = (
        semantic_gap_policy == "execute_unverified_for_evolution"
        and continuation.eligibility is ContinuationEligibility.ELIGIBLE_FOR_EVOLUTION
    )
    semantic_evolution_requested = (
        semantic_gap_policy == "evolve_before_execution"
        and continuation.eligibility is ContinuationEligibility.ELIGIBLE_FOR_EVOLUTION
    )
    continuation_receipt = None
    if continuation_authorized or semantic_evolution_requested:
        assert continuation.current_state_id is not None
        assert continuation.target_state_id is not None
        assert continuation.semantic_guard_gap is not None
        assert continuation.bound_action is not None
        continuation_receipt = SemanticGapContinuationReceipt(
            proposal_id=proposal.proposal_id,
            binding_revision=frozen_bindings.revision,
            efsm_revision=fsm.efsm_revision,
            source_state_id=continuation.current_state_id,
            target_state_id=continuation.target_state_id,
            semantic_guard_gap=continuation.semantic_guard_gap,
            bound_action=continuation.bound_action,
        ).model_dump(mode="json")

    # Topology continuation is a SEPARATE branch that runs only when the semantic-gap
    # continuation is ineligible (their verifier reasons are disjoint, so at most one
    # fires). The semantic branch above is untouched. A topology gap authorizes one
    # unverified evidence execution only for an auto-eligible action on a
    # conservatively refused topology gap; the verifier verdict stays DENY and
    # should_execute stays
    # False. Automatic authorization requires explicit, real foreground context (never
    # FSM-inferred): the receipt pins it and the topology_evolution mode revalidates it
    # against the pre-observation.
    #
    # Both non-blocking policies may authorize it. ``evolve_before_execution`` has no
    # pre-execution repair for a *topology* gap — there is nothing to enrich when the
    # screen is not in the model at all — so without this the recovery ``navigate_back``
    # that unlocalized-screen guidance suggests is itself blocked, leaving the agent in a
    # dead end the guidance cannot release. Eligibility is unchanged; only the set of
    # policies that may authorize an already-eligible action widens.
    topology_continuation_authorized = False
    topology_continuation_receipt = None
    # Why a topology gap was NOT turned into evidence. Until this record existed a
    # refusal was invisible in the episode rows: an ambiguous identity, a component
    # action, or a missing foreground all looked identical to "nothing happened".
    topology_continuation_refusal: dict[str, Any] | None = None
    if not should_execute and continuation.eligibility is ContinuationEligibility.INELIGIBLE:
        if semantic_gap_policy not in _TOPOLOGY_CONTINUATION_POLICIES:
            topology_continuation_refusal = {
                "refusal_reason": ContinuationRefusal.POLICY_GATED.value,
                "authorization": TopologyAuthorization.REJECTED.value,
                "details": (
                    f"semantic gap policy {semantic_gap_policy!r} authorizes no topology "
                    "evidence execution"
                ),
                "source_kind": None,
            }
        else:
            topo_engine = RuntimeEvolutionEngine(fsm, {})
            topology = TopologyContinuationClassifier(fsm, topo_engine).classify(
                final, screen, proposed
            )
            if (
                topology.authorization is TopologyAuthorization.AUTHORIZED_AUTO
                and topology.bound_action is not None
                and foreground_package
                and foreground_activity
            ):
                topology_continuation_authorized = True
                topology_continuation_receipt = TopologyContinuationReceipt(
                    proposal_id=proposal.proposal_id,
                    binding_revision=frozen_bindings.revision,
                    efsm_revision=fsm.efsm_revision,
                    verifier_result=_verdict(final.result),
                    verifier_reason=final.reason.value,
                    authorization_basis=str(topology.source_kind.value),
                    source_state_id=topology.current_state_id,
                    bound_action=topology.bound_action,
                    foreground_package=foreground_package,
                    foreground_activity=foreground_activity,
                ).model_dump(mode="json")
            else:
                authorized_but_blind = (
                    topology.authorization is TopologyAuthorization.AUTHORIZED_AUTO
                )
                refusal_code = (
                    ContinuationRefusal.FOREGROUND_CONTEXT_MISSING
                    if authorized_but_blind
                    else topology.refusal_code or ContinuationRefusal.NOT_TOPOLOGY_GAP
                )
                topology_continuation_refusal = {
                    "refusal_reason": refusal_code.value,
                    "authorization": topology.authorization.value,
                    "details": (
                        "no real foreground context was observed for the authorized action"
                        if authorized_but_blind
                        else topology.details
                    ),
                    "source_kind": (
                        topology.source_kind.value if topology.source_kind is not None else None
                    ),
                }
    stage_timings_ms["continuation_classification_ms"] = (time.perf_counter() - _stage_t0) * 1000.0

    _stage_t0 = time.perf_counter()
    blocked_feedback = (
        None
        if should_execute
        else _build_execution_feedback(
            fsm,
            screen,
            result,
            frozen_bindings,
            last_localized_state_id=last_localized_state_id,
        )
    )
    # A proven guard failure on an active BOUND slot authorizes an evidence-triggered
    # rebind of that slot only. This is harness control data (the ledger arms the
    # chain from it); it is never a repair option injected into the LLM prompt.
    recovery_authorization = None
    if (
        final.reason is VerifyReason.GUARD_FAILED
        and result.pre_state_id is not None
        and final.clause_activation is not None
    ):
        authorization = build_recovery_authorization(
            event_id=(
                blocked_feedback["event_id"]
                if blocked_feedback is not None
                else _event_id(result.pre_state_id, final.reason.value, "DENY", {})
            ),
            efsm_revision=fsm.efsm_revision,
            source_state_id=result.pre_state_id,
            failed_action=result.bound_action,
            activation=final.clause_activation,
            witness=final.predicate_witness,
            snapshot=frozen_bindings,
        )
        if authorization is not None:
            recovery_authorization = authorization.model_dump(mode="json")
            if blocked_feedback is not None:
                # Evidence-only affordance: names the rebindable slot and where its
                # value may legitimately come from. The attempt budget stays with the
                # harness ledger, which owns whether the channel is actually offered.
                blocked_feedback["rebind_affordance"] = _rebind_affordance(
                    authorization,
                    final.predicate_witness,
                    final.clause_activation,
                )
    stage_timings_ms["evidence_feedback_build_ms"] = (time.perf_counter() - _stage_t0) * 1000.0

    return {
        "stage_timings_ms": stage_timings_ms,
        "verdict": _verdict(final.result),
        # Same verdict, different evidentiary status: True only when a guard or an
        # invariant concretely evaluated false, or an observed successor concretely
        # mismatched. This is what lets proof-backed prevention be reported apart
        # from conservative over-refusal; it gates nothing.
        "proven_refutation": final.proven_refutation,
        "reason": final.reason.value,
        "details": final.details,
        "proposal_id": proposal.proposal_id,
        "efsm_revision": proposal.efsm_revision,
        "pre_state_id": result.pre_state_id,
        "expected_target_state_id": result.expected_target_state_id,
        "binding_status": result.binding_status.value,
        "binding_snapshot": frozen_bindings.model_dump(mode="json"),
        "clause_activation": (
            final.clause_activation.model_dump(mode="json")
            if final.clause_activation is not None
            else None
        ),
        "bound_action": result.bound_action,
        "verification_latency_ms": verification_latency_ms,
        "should_execute": should_execute,
        "execution_disposition": (
            "VERIFIED_EXECUTION"
            if should_execute
            else "EXECUTE_UNVERIFIED_FOR_EVOLUTION"
            if continuation_authorized
            else "EVOLVE_BEFORE_EXECUTION"
            if semantic_evolution_requested
            else "EXECUTE_UNVERIFIED_FOR_TOPOLOGY_EVOLUTION"
            if topology_continuation_authorized
            else "BLOCK"
        ),
        "continuation_authorized": continuation_authorized,
        "semantic_evolution_requested": semantic_evolution_requested,
        "continuation_receipt": continuation_receipt,
        "topology_continuation_authorized": topology_continuation_authorized,
        "topology_continuation_receipt": topology_continuation_receipt,
        "topology_continuation_refusal": topology_continuation_refusal,
        "semantic_guard_gap": (
            final.semantic_guard_gap.value if final.semantic_guard_gap is not None else None
        ),
        "recovery_authorization": recovery_authorization,
        "vigil_feedback": blocked_feedback,
    }


# --- post-action check ---------------------------------------------------
#
# The post-action bridge request carries only the landed screen, its foreground
# context, and the expected successor. Deriving an evolution witness from it needs the
# operands of the step that just executed (pre-screen, bound action, localized source)
# plus the episode's semantic-gap policy. None of those are added to the request: the
# bridge runs as one persistent server per runner process, so the execution call that
# immediately precedes the post-action check leaves them here, server-side.


class _ExecutionContext(BaseModel):
    """Single-use memo of the last actually-executed action, for post-action witnesses.

    Recorded for a verified ALLOW and for a typed unverified continuation
    (``EXECUTE_UNVERIFIED_FOR_[TOPOLOGY_]EVOLUTION``) -- any response after which an
    Android action really runs. The M3A runner today issues post-action checks only
    after verified executions, so the continuation memos are simply overwritten by
    the next execution; keeping them means a post-action check that does follow an
    unverified execution finds its witness operands instead of a cleared memo. It is
    pinned to the FSM file, the EFSM revision, and the expected successor, so a memo
    that does not belong to the post-action call at hand is refused rather than
    guessed at.
    """

    model_config = ConfigDict(extra="forbid")

    fsm_path: str
    semantic_gap_policy: str
    proposal_id: str
    efsm_revision: int
    pre_state_id: str
    expected_target_state_id: str | None
    bound_action: dict[str, Any]
    pre_screen: RawScreen


_LAST_EXECUTION_CONTEXT: _ExecutionContext | None = None


def _remember_execution_context(
    fsm_path: str,
    fsm: AppFSM,
    screen: RawScreen,
    response: dict[str, Any],
    semantic_gap_policy: str,
) -> None:
    """Record (or clear) the operands the next post-action check may reuse."""
    global _LAST_EXECUTION_CONTEXT
    bound_action = response.get("bound_action")
    pre_state_id = response.get("pre_state_id")
    proposal_id = response.get("proposal_id")
    disposition = response.get("execution_disposition")
    # The memo follows actual execution, not authorization: a verified ALLOW and a
    # typed unverified continuation (EXECUTE_UNVERIFIED_FOR_[TOPOLOGY_]EVOLUTION)
    # both run an Android action, so both leave witness operands. Only a step that
    # truly blocks clears the memo -- previously the should_execute gate dropped the
    # continuation witnesses too.
    will_execute = response.get("should_execute") is True or (
        isinstance(disposition, str) and disposition.startswith("EXECUTE_UNVERIFIED")
    )
    if (
        not will_execute
        or not isinstance(bound_action, dict)
        or not bound_action
        or not isinstance(pre_state_id, str)
        or not isinstance(proposal_id, str)
    ):
        # A blocked or unbound proposal is never a witness operand; clearing keeps a
        # stale memo from being matched to a later post-action call.
        _LAST_EXECUTION_CONTEXT = None
        return
    expected_target = response.get("expected_target_state_id")
    _LAST_EXECUTION_CONTEXT = _ExecutionContext(
        fsm_path=fsm_path,
        semantic_gap_policy=semantic_gap_policy,
        proposal_id=proposal_id,
        efsm_revision=fsm.efsm_revision,
        pre_state_id=pre_state_id,
        expected_target_state_id=(expected_target if isinstance(expected_target, str) else None),
        bound_action=dict(bound_action),
        pre_screen=screen,
    )


def _take_execution_context(
    fsm: AppFSM,
    fsm_path: str | None,
    expected_target_state_id: str | None,
) -> tuple[_ExecutionContext | None, ContinuationRefusal | None, str]:
    """Consume the memo iff it provably belongs to this post-action check."""
    global _LAST_EXECUTION_CONTEXT
    context = _LAST_EXECUTION_CONTEXT
    _LAST_EXECUTION_CONTEXT = None  # single use, whatever happens next
    if context is None or fsm_path is None:
        return (
            None,
            ContinuationRefusal.NO_EXECUTION_CONTEXT,
            ("no verified execution precedes this post-action check in this bridge process"),
        )
    if context.fsm_path != fsm_path:
        return None, ContinuationRefusal.EXECUTION_CONTEXT_MISMATCH, "execution used another EFSM"
    if context.efsm_revision != fsm.efsm_revision:
        return (
            None,
            ContinuationRefusal.EXECUTION_CONTEXT_MISMATCH,
            f"execution saw EFSM revision {context.efsm_revision}, now {fsm.efsm_revision}",
        )
    if context.expected_target_state_id != expected_target_state_id:
        return (
            None,
            ContinuationRefusal.EXECUTION_CONTEXT_MISMATCH,
            "post-action check reports another expected successor than the execution did",
        )
    return context, None, ""


def _resolve_exploration_trace(fsm_path: str | None) -> Path | None:
    """Resolve the exploration trace exactly, never by guessing.

    ``$VIGIL_EVOLUTION_TRACE_PATH`` takes precedence. Otherwise, the ``trace``
    field in the EFSM bundle's ``summary.json`` identifies its source evidence.
    """
    env_trace = os.environ.get("VIGIL_EVOLUTION_TRACE_PATH")
    if env_trace and Path(env_trace).expanduser().is_file():
        return Path(env_trace).expanduser()
    if fsm_path is None:
        return None
    summary_path = Path(fsm_path).parent / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    trace = summary.get("trace") if isinstance(summary, dict) else None
    if not isinstance(trace, str) or not trace:
        return None
    candidate = Path(trace).expanduser()
    if not candidate.is_absolute():
        candidate = summary_path.parent / candidate
    return candidate if candidate.is_file() else None


# One parsed exploration trace per bridge process. The trace file is immutable for the
# life of an episode, so parsing it once keeps the witness path off the critical path of
# every later post-action check; persisted runtime captures are re-read each time
# because they grow as evidence is admitted.
_TRACE_SCREEN_CACHE: dict[str, dict[str, RawScreen]] = {}


def _post_action_evidence_store(fsm_path: str | None) -> dict[str, RawScreen]:
    """Hydrate the raw-screen store a refinement needs, without a new request field.

    A store that cannot be completed simply lacks observations: the witness classifier
    then refuses attribution rather than assuming an observation is novel.
    """
    trace_path = _resolve_exploration_trace(fsm_path)
    trace_store: dict[str, RawScreen] = {}
    if trace_path is not None:
        key = str(trace_path.resolve())
        cached = _TRACE_SCREEN_CACHE.get(key)
        if cached is None:
            try:
                cached = {
                    screen.screen_id: screen
                    for screen in (
                        normalize_trace_screen(value)
                        for value in _load_evolution_trace_screens(key).values()
                    )
                }
            except (OSError, ValueError) as exc:
                print(
                    f"[vigil] post-action evidence: unusable trace {key}: {exc}",
                    file=sys.stderr,
                )
                cached = {}
            _TRACE_SCREEN_CACHE[key] = cached
        trace_store = cached
    try:
        return {**trace_store, **load_persisted_runtime_screens()}
    except ValueError:
        # No run-artifacts dir (CLI / offline replay): the trace alone is the store.
        return dict(trace_store)


def _admit_post_action_witness(
    fsm: AppFSM,
    fsm_path: str | None,
    context: _ExecutionContext,
    witness: PostActionWitness,
    post_screen: RawScreen,
    raw_screens: dict[str, RawScreen],
) -> dict[str, Any]:
    """Submit one accepted witness to the existing observed-transition service.

    Everything conservative about the pre-action topology channel holds here: the same
    :func:`~vigil.runtime.observed_transition.apply_observed_transition` sequence, the
    same runtime-fallback origin provenance on anything admitted, no LLM, and no effect on the
    already-executed action. The response stays evidence-only.
    """
    pre_screen = context.pre_screen.model_copy(
        update={"screen_id": f"rt_pre_{raw_screen_canonical_equivalence_hash(context.pre_screen)}"}
    )
    landed = post_screen.model_copy(
        update={"screen_id": f"rt_post_{raw_screen_canonical_equivalence_hash(post_screen)}"}
    )
    result = apply_observed_transition(
        fsm, pre_screen, context.bound_action, landed, dict(raw_screens)
    )

    evidence_paths: list[str] = []
    try:
        evidence_paths = [
            str(persist_runtime_screen(result.pre_snapshot)),
            str(persist_runtime_screen(result.post_snapshot)),
        ]
    except (ValueError, OSError) as exc:
        print(f"[vigil] post-action witness evidence not persisted: {exc}", file=sys.stderr)

    # Any engine invocation is persisted: the observation is recorded in the evolution
    # log even when nothing was admitted, and a mutated graph must never be served from
    # the serve-mode cache.
    if fsm_path is not None:
        fsm.serialize_atomic(fsm_path)
    _FSM_CACHE.invalidate()

    return {
        "operation": result.operation.value if result.operation is not None else None,
        "outcome": result.outcome.value,
        "model_changed": result.model_changed,
        "provisional": result.provisional,
        "efsm_revision_before": result.efsm_revision_before,
        "efsm_revision_after": result.efsm_revision_after,
        "rejection_reason": result.rejection_reason,
        "state_admission_results": {
            "source": result.source_admission.model_dump(mode="json"),
            "target": result.target_admission.model_dump(mode="json"),
        },
        "transition_admission_result": result.transition_admission.model_dump(mode="json"),
        "pre_runtime_screen_id": result.pre_snapshot.screen_id,
        "post_runtime_screen_id": result.post_snapshot.screen_id,
        "runtime_evidence_paths": evidence_paths,
        "witness_shape": witness.shape.value if witness.shape is not None else None,
        "split_subject_state_id": witness.split_subject_state_id,
    }


def _post_action_witness_record(
    fsm: AppFSM,
    fsm_path: str | None,
    post_screen: RawScreen,
    output: VerificationOutput,
    observed: str | None,
    expected_target_state_id: str | None,
) -> dict[str, Any]:
    """Derive at most one witness from this post-action check and admit it.

    Returns the machine-readable record attached to the response. Every path returns a
    record: an attempt with its engine outcome, or a refusal with a typed reason. The
    post-action check itself is never altered — the action has already executed and this
    can only add evidence.
    """
    context, refusal, details = _take_execution_context(fsm, fsm_path, expected_target_state_id)
    if context is None:
        return _witness_refusal(refusal, details, policy=None)
    # ``block`` must stay provably evolution-free: refuse before any classification, so
    # no evolution engine is constructed and no observation is even read.
    if context.semantic_gap_policy not in _TOPOLOGY_CONTINUATION_POLICIES:
        return _witness_refusal(
            ContinuationRefusal.POLICY_GATED,
            f"semantic gap policy {context.semantic_gap_policy!r} authorizes no evolution",
            policy=context.semantic_gap_policy,
            source_state_id=context.pre_state_id,
            observed_state_id=observed,
            expected_target_state_id=expected_target_state_id,
        )

    raw_screens = _post_action_evidence_store(fsm_path)
    witness = classify_post_action_witness(
        fsm,
        raw_screens,
        pre_screen=context.pre_screen,
        pre_state_id=context.pre_state_id,
        post_screen=post_screen,
        post_output=output,
        observed_state_id=observed,
        expected_target_state_id=expected_target_state_id,
    )
    record: dict[str, Any] = {
        "attempted": False,
        "policy": context.semantic_gap_policy,
        "proposal_id": context.proposal_id,
        "witness_shape": witness.shape.value if witness.shape is not None else None,
        "refusal_reason": (
            witness.refusal_code.value if witness.refusal_code is not None else None
        ),
        "details": witness.details,
        "source_state_id": witness.source_state_id,
        "observed_state_id": witness.observed_state_id,
        "expected_target_state_id": witness.expected_target_state_id,
        "split_subject_state_id": witness.split_subject_state_id,
        "operation": None,
        "outcome": EvolutionOutcome.REJECTED.value,
        "model_changed": False,
        "provisional": False,
        "efsm_revision_before": fsm.efsm_revision,
        "efsm_revision_after": fsm.efsm_revision,
    }
    if not witness.is_witness:
        return record
    record["attempted"] = True
    record.update(
        _admit_post_action_witness(fsm, fsm_path, context, witness, post_screen, raw_screens)
    )
    return record


def _witness_refusal(
    refusal: ContinuationRefusal | None,
    details: str,
    *,
    policy: str | None,
    source_state_id: str | None = None,
    observed_state_id: str | None = None,
    expected_target_state_id: str | None = None,
) -> dict[str, Any]:
    """Uniform refusal record (no engine call, no EFSM read beyond localization)."""
    return {
        "attempted": False,
        "policy": policy,
        "proposal_id": None,
        "witness_shape": None,
        "refusal_reason": refusal.value if refusal is not None else None,
        "details": details,
        "source_state_id": source_state_id,
        "observed_state_id": observed_state_id,
        "expected_target_state_id": expected_target_state_id,
        "split_subject_state_id": None,
        "operation": None,
        "outcome": EvolutionOutcome.REJECTED.value,
        "model_changed": False,
        "provisional": False,
    }


def build_post_action_response(
    fsm: AppFSM,
    post_screen: RawScreen,
    expected_target_state_id: str | None,
    *,
    fsm_path: str | None = None,
) -> dict[str, Any]:
    """Post-action check: localize o_{t+1}, match the successor, enforce invariants.

    Runs after the shielded action has already executed. It is state-level over the
    landed screen only, so it needs neither the action nor a binding snapshot. A
    known wrong successor or a failed target invariant is a proven DENY; an
    unlocalizable landing is a conservative DENY; a match with passing invariants is ALLOW.

    Args:
        fsm: The app's verified FSM.
        post_screen: The observed post-action screen ``o_{t+1}``.
        expected_target_state_id: The successor the shielded transition pointed at
            (``expected_target_state_id`` from the execution response), or ``None``.
        fsm_path: Where ``fsm`` was loaded from. Required for an admitted witness to be
            persisted; without it the witness path still runs but nothing is written.

    Returns:
        A JSON-serializable response with the tri-state verdict, the localized
        landing, an evidence-only ``vigil_feedback`` observation, and the
        ``witness_admission`` record.

    After that verdict is computed — and never affecting it — at most one evolution
    witness is derived from the same observations and submitted to the existing
    observed-transition service. The verdict, its feedback, and the agent's next step
    are unchanged either way: this response stays evidence-only.
    """
    stage_timings_ms: dict[str, float] = {}
    monitor = RuntimeMonitor(fsm)
    _stage_t0 = time.perf_counter()
    output, observed = monitor.verify_post_action(
        post_screen, expected_target_state_id, stage_timings_out=stage_timings_ms
    )
    stage_timings_ms["post_verify_total_ms"] = (time.perf_counter() - _stage_t0) * 1000.0
    _stage_t0 = time.perf_counter()
    feedback = _build_post_action_feedback(fsm, output, observed, expected_target_state_id)
    stage_timings_ms["evidence_feedback_build_ms"] = (time.perf_counter() - _stage_t0) * 1000.0
    _stage_t0 = time.perf_counter()
    try:
        witness_admission = _post_action_witness_record(
            fsm, fsm_path, post_screen, output, observed, expected_target_state_id
        )
    except Exception as exc:  # noqa: BLE001 - post-action evidence must never break a run
        traceback.print_exc(file=sys.stderr)
        _FSM_CACHE.invalidate()
        witness_admission = _witness_refusal(
            ContinuationRefusal.WITNESS_ADMISSION_ERROR,
            f"{type(exc).__name__}: {exc}",
            policy=None,
        )
    stage_timings_ms["witness_admission_ms"] = (time.perf_counter() - _stage_t0) * 1000.0
    return {
        "stage_timings_ms": stage_timings_ms,
        "verdict": _verdict(output.result),
        "proven_refutation": output.proven_refutation,
        "reason": output.reason.value,
        "details": output.details,
        "observed_target_state_id": observed,
        "expected_target_state_id": expected_target_state_id,
        "action_executed": True,
        "vigil_feedback": feedback,
        "witness_admission": witness_admission,
    }


def _build_post_action_feedback(
    fsm: AppFSM,
    output: VerificationOutput,
    observed: str | None,
    expected_target: str | None,
) -> dict[str, Any]:
    """Evidence-only observation for a completed post-action check (o_{t+1} only)."""
    reason = output.reason.value
    if reason in ("invariant_failed", "invariant_inconclusive"):
        check: dict[str, Any] = {
            "kind": "invariant",
            "status": output.result.value,
            "details": output.details,
            "target_invariants": _invariant_summaries(fsm, output.target_state_id),
        }
    else:
        # Successor comparison. On a proven mismatch verify_post_action pins the
        # expected successor in current_state_id and the observed landing in
        # target_state_id; both equal the (expected_target, observed) pair here.
        check = {
            "kind": "state_mismatch" if output.result is VerifyResult.DENY else "successor",
            "status": output.result.value,
            "expected_state_id": expected_target,
            "observed_state_id": observed,
            "expected_label": _state_label(fsm, expected_target) or None,
            "observed_label": _state_label(fsm, observed) or None,
            "details": output.details,
        }

    return {
        "type": "vigil_verification_observation",
        "phase": "execution",
        "intervention_point": "post_action",
        "event_id": _event_id(
            observed, reason, _verdict(output.result), {"expected_target": expected_target}
        ),
        "action_executed": True,
        "verdict": _verdict(output.result),
        "proven_refutation": output.proven_refutation,
        "reason": reason,
        "current_state": {
            "status": "localized" if observed is not None else "unknown",
            "state_id": observed,
            "label": _state_label(fsm, observed) or None,
        },
        "verification_evidence": {"checks": [check]},
        "process_constraints": list(_POST_ACTION_PROCESS_CONSTRAINTS),
    }


# --- semantic-gap evolution stage --------------------------------------


def _model_inventory_url(base_url: str) -> str:
    """Return the standard OpenAI-compatible model inventory endpoint."""
    return f"{base_url.rstrip('/')}/models"


def _build_evolution_llm(
    model: str,
    base_url: str,
    models_url: str,
) -> LlmClient:
    """Create the explicit offline-generation client for one evolution stage."""
    # Credentials are inherited from the configured environment.
    return LlmClient(
        LLMConfig(
            provider="openai_compatible",
            base_url=base_url,
            models_url=models_url,
            model=model,
            temperature=0.0,
        )
    )


def _load_evolution_trace_screens(trace_path: str) -> dict[str, Any]:
    trace = Path(trace_path).expanduser().resolve()
    payload = json.loads(trace.read_text(encoding="utf-8"))
    screens = payload.get("screens")
    if not isinstance(screens, dict):
        raise ValueError("evolution trace must contain a 'screens' object")
    resolved: dict[str, Any] = {}
    for screen_id, value in screens.items():
        if not isinstance(value, dict):
            raise ValueError(f"trace screen {screen_id!r} must be an object")
        screen = dict(value)
        for field in ("screenshot_path", "xml_tree_path"):
            raw_path = screen.get(field)
            if not isinstance(raw_path, str) or not raw_path:
                continue
            artifact = Path(raw_path).expanduser()
            if artifact.is_absolute():
                screen[field] = str(artifact)
                continue
            for ancestor in trace.parents:
                candidate = ancestor / artifact
                if candidate.exists():
                    screen[field] = str(candidate.resolve())
                    break
        resolved[screen_id] = screen
    return resolved


def _contract_enrichment_packet(
    fsm: AppFSM,
    pre_screen: RawScreen,
    post_screen: RawScreen | None,
    m3a_action: dict[str, Any],
    admitted_proposal: AdmittedPlannerProposal | dict[str, Any],
    continuation_receipt: SemanticGapContinuationReceipt | dict[str, Any],
) -> tuple[SemanticGapEvidencePacket, SemanticGapContinuationReceipt]:
    """Revalidate the frozen proposal, receipt, live action, and current gap."""
    proposal = _coerce_admitted_proposal(admitted_proposal)
    receipt = SemanticGapContinuationReceipt.model_validate(continuation_receipt)
    if receipt.proposal_id != proposal.proposal_id:
        raise ValueError("Contract-enrichment receipt does not match the planner proposal")
    if receipt.binding_revision != proposal.binding_snapshot.revision:
        raise ValueError("Contract-enrichment receipt has a different binding revision")
    if receipt.efsm_revision != proposal.efsm_revision:
        raise ValueError("Contract-enrichment receipt has a different EFSM revision")
    if receipt.efsm_revision != fsm.efsm_revision:
        raise ValueError("Contract-enrichment receipt is stale for the current EFSM")

    canonical_proposal = canonical_agent_action_proposal(_opaque_m3a_action_proposal(m3a_action))
    if canonical_proposal != proposal.agent_action_proposal:
        raise ValueError("Contract-enrichment action does not match the planner proposal")
    proposed = convert_m3a_action(m3a_action)
    if proposed is None:
        raise ValueError("Contract-enrichment action has no Vigil executable form")
    binding = ActionBinder().bind(proposed, pre_screen)
    if binding.status is not BindingStatus.MATCH:
        raise ValueError(
            "Contract-enrichment action does not bind uniquely on the pre-screen: "
            f"{binding.status.value}"
        )
    if canonical_action_key(binding.bound_action) != canonical_action_key(receipt.bound_action):
        raise ValueError("Contract-enrichment action differs from the verified bound action")

    lookup = fsm.resolve_transition(receipt.source_state_id, binding.bound_action)
    if lookup.status is not TransitionLookupStatus.MATCH or lookup.transition is None:
        raise ValueError("Contract-enrichment receipt no longer resolves one transition")
    if lookup.transition.target != receipt.target_state_id:
        raise ValueError("Contract-enrichment receipt target no longer matches the EFSM")
    current_gap = semantic_guard_coverage_gap(lookup.transition.guard_contract)
    if current_gap is not receipt.semantic_guard_gap:
        raise ValueError("semantic guard coverage gap changed after execution verification")

    packet = SemanticGapEvidencePacket(
        pre_screen=pre_screen,
        post_screen=post_screen,
        proposal=proposal,
        bound_action=binding.bound_action,
    )
    return packet, receipt


def build_contract_enrichment_response(
    fsm: AppFSM,
    pre_screen: RawScreen,
    m3a_action: dict[str, Any],
    admitted_proposal: AdmittedPlannerProposal | dict[str, Any],
    continuation_receipt: SemanticGapContinuationReceipt | dict[str, Any],
    raw_screens: dict[str, Any],
    llm: LlmClient,
    *,
    model: str,
    base_url: str,
    post_screen: RawScreen | None = None,
) -> dict[str, Any]:
    """Apply Contract enrichment outside execution-time verification."""
    packet, receipt = _contract_enrichment_packet(
        fsm,
        pre_screen,
        post_screen,
        m3a_action,
        admitted_proposal,
        continuation_receipt,
    )
    measured_llm = _MeasuredEvolutionLlm(llm)
    started = time.perf_counter()
    result = RuntimeEvolutionEngine(fsm, raw_screens).enrich_transition_contract(
        packet,
        measured_llm,
    )
    evolution_latency_ms = (time.perf_counter() - started) * 1000.0
    action_executed = post_screen is not None
    reason = "semantic_contract_admitted" if result.evolved else result.reason
    process_constraints = (
        [
            "The checked Android action already ran without a verified ALLOW.",
            "This remains an unverified exposure even if EFSM evolution succeeds.",
            "The evolved guard applies only to future planner proposals.",
        ]
        if action_executed
        else [
            "No Android action ran for this Contract-enrichment attempt.",
            "The original execution verdict remains DENY.",
            "A new planner proposal is required after any EFSM revision change.",
        ]
    )
    response = {
        "type": "vigil_semantic_evolution_result",
        "proposal_id": packet.proposal.proposal_id,
        "binding_revision": packet.proposal.binding_snapshot.revision,
        "semantic_guard_gap": receipt.semantic_guard_gap.value,
        "action_executed": action_executed,
        "evidence_recorded": result.evidence_recorded,
        "evolved": result.evolved,
        "outcome": result.outcome.value,
        "efsm_revision_before": result.efsm_revision_before,
        "efsm_revision_after": result.efsm_revision_after,
        "source_state_id": result.source_state_id,
        "target_state_id": result.target_state_id,
        "transition_index": result.transition_index,
        "reason": reason,
        "promotion": (
            result.promotion.model_dump(mode="json") if result.promotion is not None else None
        ),
        "llm_model": model,
        "llm_base_url": base_url,
        "llm_call_count": measured_llm.call_count,
        "llm_latency_ms": measured_llm.latency_ms,
        "evolution_latency_ms": evolution_latency_ms,
    }
    response["vigil_feedback"] = {
        "type": "vigil_verification_observation",
        "phase": "semantic_evolution",
        "intervention_point": (
            "post_unverified_execution" if action_executed else "pre_action_evolution"
        ),
        "action_executed": action_executed,
        "verdict": "OBSERVE",
        "reason": reason or "semantic_evolution_not_admitted",
        "efsm_revision_before": result.efsm_revision_before,
        "efsm_revision_after": result.efsm_revision_after,
        "verification_evidence": {
            "checks": [
                {
                    "kind": "semantic_guard_coverage",
                    "gap": response["semantic_guard_gap"],
                    "status": "evolved" if result.evolved else "not_admitted",
                    "details": result.reason,
                }
            ]
        },
        "process_constraints": process_constraints,
    }
    return response


def _evolution_artifact_dir(kind: str, proposal_id: str, efsm_revision: int) -> Path:
    """Fresh per-attempt debug-dump directory under VIGIL_RUN_ARTIFACTS_DIR.

    The sha256 key only makes the proposal id filesystem-safe; attempts at the same
    proposal/revision are numbered by counting existing attempt directories.
    """
    root = os.environ.get("VIGIL_RUN_ARTIFACTS_DIR")
    if not root:
        raise ValueError(f"{kind} evolution requires VIGIL_RUN_ARTIFACTS_DIR")
    key = hashlib.sha256(f"{proposal_id}:{efsm_revision}".encode()).hexdigest()[:16]
    base = Path(root) / kind / key
    base.mkdir(parents=True, exist_ok=True)
    attempt = sum(
        1 for child in base.iterdir() if child.is_dir() and child.name.startswith("attempt_")
    )
    artifact_dir = base / f"attempt_{attempt:03d}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def _handle_contract_enrichment_request(
    fsm: AppFSM,
    fsm_path: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    pre_xml = request.get("pre_screen_xml")
    if not isinstance(pre_xml, str):
        raise ValueError("semantic_evolution mode requires 'pre_screen_xml'")
    post_xml = request.get("post_screen_xml")
    if post_xml is not None and not isinstance(post_xml, str):
        raise ValueError("semantic_evolution 'post_screen_xml' must be a string")
    action = request.get("action")
    proposal_payload = request.get("admitted_proposal")
    receipt_payload = request.get("continuation_receipt")
    if not isinstance(action, dict):
        raise ValueError("semantic_evolution mode requires an 'action' object")
    if not isinstance(proposal_payload, dict):
        raise ValueError("semantic_evolution mode requires 'admitted_proposal'")
    if not isinstance(receipt_payload, dict):
        raise ValueError("semantic_evolution mode requires 'continuation_receipt'")
    model = request.get("evolution_model")
    base_url = request.get("evolution_base_url")
    trace_path = request.get("evolution_trace_path")
    if not isinstance(model, str) or not model:
        raise ValueError("semantic_evolution mode requires 'evolution_model'")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("semantic_evolution mode requires 'evolution_base_url'")
    if not isinstance(trace_path, str) or not trace_path:
        raise ValueError("semantic_evolution mode requires 'evolution_trace_path'")

    proposal = _coerce_admitted_proposal(proposal_payload)
    artifact_dir = _evolution_artifact_dir(
        "contract_enrichment",
        proposal.proposal_id,
        proposal.efsm_revision,
    )
    pre_path = artifact_dir / "pre.xml"
    post_path = artifact_dir / "post.xml" if post_xml is not None else None
    pre_path.write_text(pre_xml, encoding="utf-8")
    if post_path is not None:
        post_path.write_text(post_xml, encoding="utf-8")
    request_record = {
        "proposal": proposal_payload,
        "continuation_receipt": receipt_payload,
        "action": action,
        "evolution_model": model,
        "evolution_base_url": base_url,
        "evolution_trace_path": trace_path,
        "pre_xml_path": str(pre_path),
        "post_xml_path": str(post_path) if post_path is not None else None,
    }
    (artifact_dir / "request.json").write_text(
        json.dumps(request_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    app_package = request.get("app_package")
    # Prefer the runner's observed foreground over an FSM-inferred activity.
    # Contract admission re-localizes the source observation, so its observed
    # activity must remain part of the structural fingerprint. Activity inference
    # is used only when the caller did not provide foreground context.
    foreground_package = request.get("pre_foreground_package")
    foreground_activity = request.get("pre_foreground_activity")
    if isinstance(foreground_package, str) and foreground_package:
        activity_name = foreground_activity if isinstance(foreground_activity, str) else None
        package_name = foreground_package
    else:
        activity_name = _infer_activity_name(fsm)
        package_name = None
    pre_screen = raw_screen_from_xml(
        pre_xml,
        screen_id="runtime",
        app_package=app_package,
        activity_name=activity_name,
        package_name=package_name,
    )
    post_screen = (
        raw_screen_from_xml(
            post_xml,
            screen_id="runtime",
            app_package=app_package,
            activity_name=activity_name,
            package_name=package_name,
        )
        if post_xml is not None
        else None
    )
    raw_screens = _load_evolution_trace_screens(trace_path)
    models_url = _model_inventory_url(base_url)
    measured_events_start = len(fsm.evolution_log)
    response = build_contract_enrichment_response(
        fsm,
        pre_screen,
        action,
        proposal,
        receipt_payload,
        raw_screens,
        _build_evolution_llm(model, base_url, models_url),
        model=model,
        base_url=base_url,
        post_screen=post_screen,
    )
    response.update(
        {
            "artifact_dir": str(artifact_dir),
            "pre_xml_path": str(pre_path),
            "post_xml_path": str(post_path) if post_path is not None else None,
            "result_path": str(artifact_dir / "result.json"),
            "evolution_trace_path": trace_path,
        }
    )
    for event in fsm.evolution_log[measured_events_start:]:
        event.update(
            {
                "artifact_dir": str(artifact_dir),
                "evolution_trace_path": trace_path,
                "llm_model": model,
                "llm_call_count": response["llm_call_count"],
                "llm_latency_ms": response["llm_latency_ms"],
            }
        )
    if len(fsm.evolution_log) > measured_events_start:
        fsm.serialize(fsm_path)
    Path(response["result_path"]).write_text(
        json.dumps(response, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return response


def _context_screen(
    xml: str, app_package: str | None, package_name: str, activity_name: str, screen_id: str
) -> RawScreen:
    """Build a RawScreen with its own real foreground context (never FSM-inferred)."""
    return raw_screen_from_xml(
        xml,
        screen_id=screen_id,
        app_package=app_package,
        activity_name=activity_name or None,
        package_name=package_name,
    )


def _observed_transition_packet(
    fsm: AppFSM,
    pre_screen: RawScreen,
    action: dict[str, Any],
    admitted_proposal: dict[str, Any],
    receipt_payload: dict[str, Any],
) -> tuple[AdmittedPlannerProposal, dict[str, Any], TopologyContinuationReceipt]:
    """Revalidate the topology receipt against the pre-observation.

    This runs AFTER the action executed, so it cannot prevent the exposure — it binds
    admission to exactly the authorized proposal, action, pre-observation, and foreground
    context. The post-screen is deliberately NOT constrained here: an action that
    legitimately left the app is a real observed outcome the generic service records and
    rejects factually, not a receipt violation.
    """
    proposal = _coerce_admitted_proposal(admitted_proposal)
    receipt = TopologyContinuationReceipt(**receipt_payload)

    if receipt.proposal_id != proposal.proposal_id:
        raise ValueError("topology receipt proposal_id does not match the admitted proposal")
    if receipt.binding_revision != proposal.binding_snapshot.revision:
        raise ValueError("topology receipt binding_revision is stale")
    if receipt.efsm_revision != proposal.efsm_revision:
        raise ValueError("topology receipt efsm_revision does not match the proposal")
    if proposal.efsm_revision != fsm.efsm_revision:
        raise ValueError(
            f"topology proposal uses EFSM revision {proposal.efsm_revision}; "
            f"current revision is {fsm.efsm_revision}"
        )

    canonical_action = canonical_agent_action_proposal(_opaque_m3a_action_proposal(action))
    if canonical_action != proposal.agent_action_proposal:
        raise ValueError("topology action does not match the admitted planner proposal")

    proposed = convert_m3a_action(action)
    if proposed is None:
        raise ValueError("topology action has no Vigil executable form")

    binding = ActionBinder().bind(proposed, pre_screen)
    if binding.status is not BindingStatus.MATCH:
        raise ValueError(f"topology action did not bind to the pre-screen: {binding.status.value}")
    bound_action = binding.bound_action

    if bound_action != receipt.bound_action:
        raise ValueError("freshly rebound action differs from the receipt bound action")

    # The pre-observation must be in-app; the post-screen is unconstrained (see docstring).
    if pre_screen.package_name != fsm.app_package:
        raise ValueError(
            f"pre-screen foreground package {pre_screen.package_name!r} is not {fsm.app_package!r}"
        )
    if receipt.foreground_package != pre_screen.package_name:
        raise ValueError("receipt foreground package does not match the pre-observation")
    if receipt.foreground_activity != pre_screen.activity_name:
        raise ValueError("receipt foreground activity does not match the pre-observation")

    # Reconfirm the source is still an authorized topology source on this pre-observation.
    verification = VerificationOutput(
        result=VerifyResult(receipt.verifier_result.lower()),
        reason=VerifyReason(receipt.verifier_reason),
        current_state_id=receipt.source_state_id,
    )
    topo_engine = RuntimeEvolutionEngine(fsm, {})
    reconfirm = TopologyContinuationClassifier(fsm, topo_engine).classify(
        verification, pre_screen, proposed
    )
    if reconfirm.authorization is not TopologyAuthorization.AUTHORIZED_AUTO:
        raise ValueError(
            f"pre-observation is no longer an authorized topology source: {reconfirm.details}"
        )
    if reconfirm.source_kind is None or reconfirm.source_kind.value != receipt.authorization_basis:
        raise ValueError("topology source classification no longer matches the receipt basis")
    # Exact source-state identity: for a runtime-evolved source the reclassified state id
    # must equal the receipt's (comparing only source kind is insufficient); for a novel
    # source both are None.
    if reconfirm.current_state_id != receipt.source_state_id:
        raise ValueError("reclassified source state id does not match the receipt")

    return proposal, bound_action, receipt


def _handle_observed_transition_request(
    fsm: AppFSM,
    fsm_path: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Post-execution topology admission bridge mode (no LLM).

    Records the executed ``<pre, action, post>``, admits/corroborates/refines topology
    via the agent-independent service, persists the run-local evidence and EFSM
    atomically, and returns a factual ``vigil_topology_evolution_result``.
    """
    pre_xml = request.get("pre_screen_xml")
    post_xml = request.get("post_screen_xml")
    action = request.get("action")
    proposal_payload = request.get("admitted_proposal")
    receipt_payload = request.get("topology_continuation_receipt")
    trace_path = request.get("evolution_trace_path")
    pre_pkg = request.get("pre_foreground_package")
    pre_act = request.get("pre_foreground_activity")
    post_pkg = request.get("post_foreground_package")
    post_act = request.get("post_foreground_activity")
    if not isinstance(pre_xml, str):
        raise ValueError("topology_evolution mode requires 'pre_screen_xml'")
    if not isinstance(post_xml, str):
        raise ValueError("topology_evolution mode requires 'post_screen_xml'")
    if not isinstance(action, dict):
        raise ValueError("topology_evolution mode requires an 'action' object")
    if not isinstance(proposal_payload, dict):
        raise ValueError("topology_evolution mode requires 'admitted_proposal'")
    if not isinstance(receipt_payload, dict):
        raise ValueError("topology_evolution mode requires 'topology_continuation_receipt'")
    if not isinstance(trace_path, str) or not trace_path:
        raise ValueError("topology_evolution mode requires 'evolution_trace_path'")
    for name, value in (
        ("pre_foreground_package", pre_pkg),
        ("pre_foreground_activity", pre_act),
        ("post_foreground_package", post_pkg),
        ("post_foreground_activity", post_act),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"topology_evolution mode requires '{name}'")

    pre_screen = _context_screen(pre_xml, fsm.app_package, pre_pkg, pre_act, "topo_pre")
    pre_hash = raw_screen_canonical_equivalence_hash(pre_screen)
    pre_screen = pre_screen.model_copy(update={"screen_id": f"rt_pre_{pre_hash}"})
    post_screen = _context_screen(post_xml, fsm.app_package, post_pkg, post_act, "topo_post")
    post_hash = raw_screen_canonical_equivalence_hash(post_screen)
    post_screen = post_screen.model_copy(update={"screen_id": f"rt_post_{post_hash}"})

    proposal, bound_action, receipt = _observed_transition_packet(
        fsm, pre_screen, action, proposal_payload, receipt_payload
    )

    artifact_dir = _evolution_artifact_dir(
        "observed_transition", proposal.proposal_id, proposal.efsm_revision
    )
    (artifact_dir / "pre.xml").write_text(pre_xml, encoding="utf-8")
    (artifact_dir / "post.xml").write_text(post_xml, encoding="utf-8")
    (artifact_dir / "request.json").write_text(
        json.dumps(
            {
                "proposal": proposal_payload,
                "topology_continuation_receipt": receipt_payload,
                "action": action,
                "evolution_trace_path": trace_path,
                "pre_foreground_package": pre_pkg,
                "pre_foreground_activity": pre_act,
                "post_foreground_package": post_pkg,
                "post_foreground_activity": post_act,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    trace_screens = _load_evolution_trace_screens(trace_path)
    raw_screens = load_runtime_evidence(trace_screens)

    measured_events_start = len(fsm.evolution_log)
    result = apply_observed_transition(
        fsm,
        pre_screen,
        bound_action,
        post_screen,
        raw_screens,
    )

    # Persist executed evidence regardless of admission outcome, so provenance ids stay
    # resolvable in later adapter calls; rejected admission simply attaches no edge.
    evidence_paths = [
        str(persist_runtime_screen(result.pre_snapshot)),
        str(persist_runtime_screen(result.post_snapshot)),
    ]

    response = {
        "type": "vigil_topology_evolution_result",
        "proposal_id": proposal.proposal_id,
        "binding_revision": receipt.binding_revision,
        "verifier_reason": receipt.verifier_reason,
        "authorization_basis": receipt.authorization_basis,
        "evidence_recorded": True,
        "model_changed": result.model_changed,
        "operation": result.operation.value if result.operation is not None else None,
        "outcome": result.outcome.value,
        "state_admission_results": {
            "source": result.source_admission.model_dump(mode="json"),
            "target": result.target_admission.model_dump(mode="json"),
        },
        "transition_admission_result": result.transition_admission.model_dump(mode="json"),
        "pre_runtime_screen_id": result.pre_snapshot.screen_id,
        "post_runtime_screen_id": result.post_snapshot.screen_id,
        "provisional": result.provisional,
        "efsm_revision_before": result.efsm_revision_before,
        "efsm_revision_after": result.efsm_revision_after,
        "action_executed": True,
        "rejection_reason": result.rejection_reason,
        "artifact_dir": str(artifact_dir),
        "pre_xml_path": str(artifact_dir / "pre.xml"),
        "post_xml_path": str(artifact_dir / "post.xml"),
        "result_path": str(artifact_dir / "result.json"),
        "evolution_trace_path": trace_path,
        "runtime_evidence_paths": evidence_paths,
        "vigil_feedback": {
            "verdict": "OBSERVE",
            "phase": "topology_evolution",
            "intervention_point": "post_unverified_execution",
            "operation": result.operation.value if result.operation is not None else None,
            "outcome": result.outcome.value,
            "process_constraints": [
                "This was an unverified topology-evolution exposure, not a verified ALLOW.",
                "Any newly admitted edge carries runtime-fallback origin provenance.",
                "The EFSM revision may have advanced; plan a fresh proposal.",
            ],
        },
    }

    for event in fsm.evolution_log[measured_events_start:]:
        event.update(
            {
                "artifact_dir": str(artifact_dir),
                "evolution_trace_path": trace_path,
            }
        )
    if len(fsm.evolution_log) > measured_events_start:
        fsm.serialize_atomic(fsm_path)
    Path(response["result_path"]).write_text(
        json.dumps(response, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return response


def _observe_witness_rejection(
    artifact_dir: Path,
    fsm: AppFSM,
    *,
    witness_index: int,
    rejection_reason: str,
) -> dict[str, Any]:
    """Factual no-evolution response for an observe witness the service cannot use."""
    response = {
        "type": "vigil_observe_witness_result",
        "witness_index": witness_index,
        "evidence_recorded": True,
        "model_changed": False,
        "operation": None,
        "outcome": "rejected",
        "provisional": False,
        "efsm_revision_before": fsm.efsm_revision,
        "efsm_revision_after": fsm.efsm_revision,
        "action_executed": True,
        "rejection_reason": rejection_reason,
        "artifact_dir": str(artifact_dir),
        "result_path": str(artifact_dir / "result.json"),
    }
    Path(response["result_path"]).write_text(
        json.dumps(response, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return response


def _handle_observe_witness_request(
    fsm: AppFSM,
    fsm_path: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Observe-mode witness admission bridge mode (no LLM, no receipt).

    The observe episode policy executes the agent's actions unshielded after the
    first UNCERTAIN and records every subsequent ``<pre, action, post>`` triple;
    the runner submits them here at episode end. Unlike ``topology_evolution``
    there is no authorization receipt to revalidate -- the action already ran on
    the agent's own responsibility and was never authorized -- so this mode is
    purely factual: bind the action against the recorded pre-observation, hand
    the triple to the agent-independent observed-transition service, and report
    the admission outcome. An unbindable action or an out-of-app observation is
    a recorded factual rejection, never an error: post-bail steps are arbitrary
    agent behavior, and the service's own admission gates are the authority on
    what enters the transition graph. Any minted edge carries runtime-fallback provenance
    and the PENDING placeholder contract, so nothing admitted here ALLOWs before
    semantic admission.
    """
    pre_xml = request.get("pre_screen_xml")
    post_xml = request.get("post_screen_xml")
    action = request.get("action")
    trace_path = request.get("evolution_trace_path")
    pre_pkg = request.get("pre_foreground_package")
    pre_act = request.get("pre_foreground_activity")
    post_pkg = request.get("post_foreground_package")
    post_act = request.get("post_foreground_activity")
    witness_index = request.get("witness_index")
    if not isinstance(pre_xml, str):
        raise ValueError("observe_witness mode requires 'pre_screen_xml'")
    if not isinstance(post_xml, str):
        raise ValueError("observe_witness mode requires 'post_screen_xml'")
    if not isinstance(action, dict):
        raise ValueError("observe_witness mode requires an 'action' object")
    if not isinstance(trace_path, str) or not trace_path:
        raise ValueError("observe_witness mode requires 'evolution_trace_path'")
    if not isinstance(witness_index, int):
        raise ValueError("observe_witness mode requires an integer 'witness_index'")
    expected_revision = request.get("expected_efsm_revision_before")
    if isinstance(expected_revision, int) and expected_revision != fsm.efsm_revision:
        # The runner tracks the revision chain across its episode-end
        # applications; a mismatch means this witness (or an earlier one) was
        # already applied and persisted by an attempt whose transport reply was
        # lost. Refuse factually rather than corroborating the same observation
        # twice.
        artifact_dir = _evolution_artifact_dir(
            "observe_witness", f"w{witness_index:03d}", fsm.efsm_revision
        )
        return _observe_witness_rejection(
            artifact_dir,
            fsm,
            witness_index=witness_index,
            rejection_reason=(
                f"revision chain mismatch: expected {expected_revision}, bundle is at "
                f"{fsm.efsm_revision}; refusing a possible duplicate application"
            ),
        )
    for name, value in (
        ("pre_foreground_package", pre_pkg),
        ("pre_foreground_activity", pre_act),
        ("post_foreground_package", post_pkg),
        ("post_foreground_activity", post_act),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"observe_witness mode requires '{name}'")

    pre_screen = _context_screen(pre_xml, fsm.app_package, pre_pkg, pre_act, "obs_pre")
    pre_hash = raw_screen_canonical_equivalence_hash(pre_screen)
    pre_screen = pre_screen.model_copy(update={"screen_id": f"rt_pre_{pre_hash}"})
    post_screen = _context_screen(post_xml, fsm.app_package, post_pkg, post_act, "obs_post")
    post_hash = raw_screen_canonical_equivalence_hash(post_screen)
    post_screen = post_screen.model_copy(update={"screen_id": f"rt_post_{post_hash}"})

    artifact_dir = _evolution_artifact_dir(
        "observe_witness", f"w{witness_index:03d}", fsm.efsm_revision
    )
    (artifact_dir / "pre.xml").write_text(pre_xml, encoding="utf-8")
    (artifact_dir / "post.xml").write_text(post_xml, encoding="utf-8")
    (artifact_dir / "request.json").write_text(
        json.dumps(
            {
                "witness_index": witness_index,
                "action": action,
                "evolution_trace_path": trace_path,
                "pre_foreground_package": pre_pkg,
                "pre_foreground_activity": pre_act,
                "post_foreground_package": post_pkg,
                "post_foreground_activity": post_act,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    proposed = convert_m3a_action(action)
    if proposed is None:
        return _observe_witness_rejection(
            artifact_dir,
            fsm,
            witness_index=witness_index,
            rejection_reason="action has no Vigil executable form",
        )
    binding = ActionBinder().bind(proposed, pre_screen)
    if binding.status is not BindingStatus.MATCH:
        return _observe_witness_rejection(
            artifact_dir,
            fsm,
            witness_index=witness_index,
            rejection_reason=f"action did not bind to the pre-observation: {binding.status.value}",
        )

    trace_screens = _load_evolution_trace_screens(trace_path)
    raw_screens = load_runtime_evidence(trace_screens)

    measured_events_start = len(fsm.evolution_log)
    result = apply_observed_transition(
        fsm,
        pre_screen,
        binding.bound_action,
        post_screen,
        raw_screens,
    )

    # Persist executed evidence regardless of admission outcome, so provenance ids stay
    # resolvable in later adapter calls; rejected admission simply attaches no edge.
    evidence_paths = [
        str(persist_runtime_screen(result.pre_snapshot)),
        str(persist_runtime_screen(result.post_snapshot)),
    ]

    response = {
        "type": "vigil_observe_witness_result",
        "witness_index": witness_index,
        "evidence_recorded": True,
        "model_changed": result.model_changed,
        "operation": result.operation.value if result.operation is not None else None,
        "outcome": result.outcome.value,
        "state_admission_results": {
            "source": result.source_admission.model_dump(mode="json"),
            "target": result.target_admission.model_dump(mode="json"),
        },
        "transition_admission_result": result.transition_admission.model_dump(mode="json"),
        "pre_runtime_screen_id": result.pre_snapshot.screen_id,
        "post_runtime_screen_id": result.post_snapshot.screen_id,
        "provisional": result.provisional,
        "efsm_revision_before": result.efsm_revision_before,
        "efsm_revision_after": result.efsm_revision_after,
        "action_executed": True,
        "rejection_reason": result.rejection_reason,
        "artifact_dir": str(artifact_dir),
        "pre_xml_path": str(artifact_dir / "pre.xml"),
        "post_xml_path": str(artifact_dir / "post.xml"),
        "result_path": str(artifact_dir / "result.json"),
        "evolution_trace_path": trace_path,
        "runtime_evidence_paths": evidence_paths,
    }

    for event in fsm.evolution_log[measured_events_start:]:
        event.update(
            {
                "artifact_dir": str(artifact_dir),
                "evolution_trace_path": trace_path,
            }
        )
    if len(fsm.evolution_log) > measured_events_start:
        fsm.serialize_atomic(fsm_path)
    Path(response["result_path"]).write_text(
        json.dumps(response, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return response


# --- JSON bridge ---------------------------------------------------------
# Modes whose handler mutates the in-memory ``AppFSM`` and only conditionally writes it
# back. They must never read or populate the serve-mode cache: a rejected evolution can
# leave partial mutations on the object that were never persisted.
#
# ``post_action`` is deliberately NOT in this set. It mutates only when a witness is
# actually submitted — rare — and that path persists the graph and invalidates the cache
# itself (``_admit_post_action_witness``), including on the error route. Listing it here
# instead would re-parse the EFSM on every landed screen and drop the cache for the next
# execution check, which is pure latency for the overwhelmingly common no-witness case.
_MUTATING_MODES = frozenset({"semantic_evolution", "topology_evolution", "observe_witness"})


class _ServeFsmCache:
    """Single-entry ``AppFSM`` cache for the ``--serve`` transport.

    Inert until :meth:`arm` is called, so the one-shot transport keeps re-parsing on
    every call exactly as before. When armed, a cached graph is reused only while the
    file's ``(path, st_mtime_ns, st_size)`` is unchanged. Runtime evolution rewrites
    ``fsm.json`` in place and ``efsm_revision`` lives *inside* the payload, so the file
    stat — not any field of the graph — is the outer guard against serving a stale
    model.
    """

    def __init__(self) -> None:
        self._armed = False
        self._store = False
        self._key: tuple[str, int, int] | None = None
        self._fsm: AppFSM | None = None

    def arm(self) -> None:
        """Enable caching for the lifetime of a serve loop."""
        self._armed = True
        self._store = True
        self.invalidate()

    def invalidate(self) -> None:
        """Drop the cached graph; the next load re-parses from disk."""
        self._key = None
        self._fsm = None

    def prepare(self, mode: object) -> None:
        """Decide whether the request about to be dispatched may use the cache."""
        self._store = self._armed and mode not in _MUTATING_MODES
        if not self._store:
            self.invalidate()

    def load(self, fsm_path: str | Path) -> AppFSM:
        """Return the FSM at ``fsm_path``, reusing the cached graph when it is fresh."""
        if not self._store:
            return AppFSM.deserialize(fsm_path)
        stat = os.stat(fsm_path)
        key = (str(fsm_path), stat.st_mtime_ns, stat.st_size)
        if key == self._key and self._fsm is not None:
            return self._fsm
        fsm = AppFSM.deserialize(fsm_path)
        self._key = key
        self._fsm = fsm
        return fsm


_FSM_CACHE = _ServeFsmCache()


# Request-scoped latency stamps (fsm load / screen parse) collected while one bridge
# request is handled. The bridge is single-threaded (serve loop and one-shot alike),
# so a module-level dict is safe. Purely observational.
_REQUEST_STAGE_TIMINGS: dict[str, float] = {}


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one bridge request. Raises SystemExit on trust-boundary errors.

    Additive latency instrumentation: every dict response gains/extends a
    ``stage_timings_ms`` mapping with ``request_total_ms`` plus any request-scoped
    stamps (``fsm_load_ms``, ``screen_parse_ms``).
    """
    total_t0 = time.perf_counter()
    _REQUEST_STAGE_TIMINGS.clear()
    response = _handle_request_inner(request)
    timings = response.setdefault("stage_timings_ms", {})
    timings.update(_REQUEST_STAGE_TIMINGS)
    timings["request_total_ms"] = (time.perf_counter() - total_t0) * 1000.0
    return response


def _handle_request_inner(request: dict[str, Any]) -> dict[str, Any]:
    mode = request.get("mode")
    if mode not in _MODES:
        raise SystemExit(f"m3a_adapter: 'mode' must be one of {_MODES}, got {mode!r}")
    fsm_path = request.get("fsm_path")
    if not fsm_path:
        raise SystemExit("m3a_adapter: 'fsm_path' is required")
    _stage_t0 = time.perf_counter()
    fsm = _FSM_CACHE.load(fsm_path)
    _REQUEST_STAGE_TIMINGS["fsm_load_ms"] = (time.perf_counter() - _stage_t0) * 1000.0

    if mode == "intent_patch":
        intent_patch_request = request.get("intent_patch_request")
        intent_patch = request.get("intent_patch")
        proposed_action = request.get("proposed_action")
        if not isinstance(intent_patch_request, dict):
            raise SystemExit("m3a_adapter: intent_patch mode requires 'intent_patch_request'")
        if not isinstance(intent_patch, dict):
            raise SystemExit("m3a_adapter: intent_patch mode requires 'intent_patch'")
        if not isinstance(request.get("binding_snapshot"), dict):
            raise SystemExit(
                "m3a_adapter: intent_patch mode requires the explicit 'binding_snapshot' base"
            )
        if not isinstance(proposed_action, dict):
            raise SystemExit(
                "m3a_adapter: intent_patch mode requires the same-round 'proposed_action'"
            )
        goal_text = request.get("goal_text")
        screen_texts = request.get("screen_texts")
        return build_intent_patch_response(
            fsm,
            intent_patch_request,
            intent_patch,
            request.get("binding_snapshot"),
            proposed_action,
            goal_text=goal_text if isinstance(goal_text, str) else None,
            screen_texts=tuple(
                text
                for text in (screen_texts if isinstance(screen_texts, list) else [])
                if isinstance(text, str)
            ),
            observe_bailed=request.get("observe_bailed") is True,
        )

    if mode == "intent_rebind":
        intent_patch = request.get("intent_patch")
        recovery_authorization = request.get("recovery_authorization")
        proposed_action = request.get("proposed_action")
        if not isinstance(intent_patch, dict):
            raise SystemExit("m3a_adapter: intent_rebind mode requires 'intent_patch'")
        if not isinstance(recovery_authorization, dict):
            raise SystemExit(
                "m3a_adapter: intent_rebind mode requires the Vigil-issued 'recovery_authorization'"
            )
        if not isinstance(request.get("binding_snapshot"), dict):
            raise SystemExit(
                "m3a_adapter: intent_rebind mode requires the explicit 'binding_snapshot' base"
            )
        if not isinstance(proposed_action, dict):
            raise SystemExit(
                "m3a_adapter: intent_rebind mode requires the same-round 'proposed_action'"
            )
        return build_intent_rebind_response(
            fsm,
            request.get("binding_snapshot"),
            intent_patch,
            recovery_authorization,
            proposed_action,
        )

    if mode == "semantic_evolution":
        return _handle_contract_enrichment_request(fsm, str(fsm_path), request)

    if mode == "topology_evolution":
        return _handle_observed_transition_request(fsm, str(fsm_path), request)

    if mode == "observe_witness":
        return _handle_observe_witness_request(fsm, str(fsm_path), request)

    if mode == "post_action":
        post_screen_xml = request.get("post_screen_xml")
        if not isinstance(post_screen_xml, str):
            raise SystemExit("m3a_adapter: post_action mode requires 'post_screen_xml' (string)")
        expected_target_state_id = request.get("expected_target_state_id")
        app_package = request.get("app_package")
        foreground_package = request.get("foreground_package")
        foreground_activity = request.get("foreground_activity")
        _stage_t0 = time.perf_counter()
        if isinstance(foreground_package, str) and foreground_package:
            post_screen = raw_screen_from_xml(
                post_screen_xml,
                screen_id="runtime",
                app_package=app_package,
                activity_name=(
                    foreground_activity if isinstance(foreground_activity, str) else None
                ),
                package_name=foreground_package,
            )
        else:
            post_screen = raw_screen_from_xml(
                post_screen_xml,
                screen_id="runtime",
                app_package=app_package,
                activity_name=_infer_activity_name(fsm),
            )
        _REQUEST_STAGE_TIMINGS["screen_parse_ms"] = (time.perf_counter() - _stage_t0) * 1000.0
        return build_post_action_response(
            fsm,
            post_screen,
            expected_target_state_id if isinstance(expected_target_state_id, str) else None,
            fsm_path=str(fsm_path),
        )

    screen_xml = request.get("screen_xml")
    if not isinstance(screen_xml, str):
        raise SystemExit("m3a_adapter: 'screen_xml' (string) is required")

    app_package = request.get("app_package")
    # Real foreground context, when the runner supplies it, is authoritative: a novel
    # screen must carry its true package/activity (never inferred from the FSM). Absent
    # foreground values fall back to the prior FSM-inferred activity so existing bridge
    # calls are unchanged.
    foreground_package = request.get("foreground_package")
    foreground_activity = request.get("foreground_activity")
    _stage_t0 = time.perf_counter()
    if isinstance(foreground_package, str) and foreground_package:
        screen = raw_screen_from_xml(
            screen_xml,
            screen_id="runtime",
            app_package=app_package,
            activity_name=(foreground_activity if isinstance(foreground_activity, str) else None),
            package_name=foreground_package,
        )
    else:
        screen = raw_screen_from_xml(
            screen_xml,
            screen_id="runtime",
            app_package=app_package,
            activity_name=_infer_activity_name(fsm),
        )
    _REQUEST_STAGE_TIMINGS["screen_parse_ms"] = (time.perf_counter() - _stage_t0) * 1000.0

    # Episode-local localization history reported by the runner. It is advisory
    # presentation context only — it never resolves a state, a transition, or a verdict —
    # so a missing or non-conforming value simply degrades to the old wording.
    last_localized_state_id = request.get("last_localized_state_id")
    if not isinstance(last_localized_state_id, str) or not last_localized_state_id:
        last_localized_state_id = None
    consecutive_unlocalized_rounds = request.get("consecutive_unlocalized_rounds")
    if not isinstance(consecutive_unlocalized_rounds, int) or isinstance(
        consecutive_unlocalized_rounds, bool
    ):
        consecutive_unlocalized_rounds = 0

    if mode == "planner":
        if not isinstance(request.get("binding_snapshot"), dict):
            raise SystemExit(
                "m3a_adapter: planner mode requires the episode's explicit 'binding_snapshot'"
            )
        return build_planner_response(
            fsm,
            screen,
            binding_snapshot=request.get("binding_snapshot"),
            live_ui_elements=request.get("ui_elements"),
            last_localized_state_id=last_localized_state_id,
            consecutive_unlocalized_rounds=max(consecutive_unlocalized_rounds, 0),
        )

    action = request.get("action")
    if not isinstance(action, dict):
        raise SystemExit("m3a_adapter: execution mode requires an 'action' object")
    if not isinstance(request.get("admitted_proposal"), dict):
        raise SystemExit(
            "m3a_adapter: execution mode requires the planner-admitted 'admitted_proposal'"
        )
    open_app_resolved_package = request.get("open_app_resolved_package")
    if not isinstance(open_app_resolved_package, str) or not open_app_resolved_package:
        open_app_resolved_package = None
    semantic_gap_policy = request.get("semantic_gap_policy", "block")
    response = build_execution_response(
        fsm,
        screen,
        action,
        admitted_proposal=request.get("admitted_proposal"),
        semantic_gap_policy=semantic_gap_policy,
        foreground_package=foreground_package if isinstance(foreground_package, str) else None,
        foreground_activity=foreground_activity if isinstance(foreground_activity, str) else None,
        last_localized_state_id=last_localized_state_id,
        open_app_resolved_package=open_app_resolved_package,
    )
    # Leave the operands of an authorized execution for the paired post-action check.
    # This adds nothing to either request and changes no response.
    _remember_execution_context(
        str(fsm_path),
        fsm,
        screen,
        response,
        semantic_gap_policy if isinstance(semantic_gap_policy, str) else "block",
    )
    return response


# --- persistent (--serve) transport ---------------------------------------
# Reserved envelope key for a request that was received and dispatched but failed inside
# the adapter. It is the serve-mode equivalent of the one-shot transport's non-zero exit,
# so the client must map it to the same exception rather than retrying the request.
SERVE_ERROR_KEY = "__vigil_bridge_error__"


def _serve_error(error: str, message: str) -> dict[str, Any]:
    """Build the structured error envelope returned instead of a response."""
    return {SERVE_ERROR_KEY: {"error": error, "message": message}}


def _serve_one(line: str) -> dict[str, Any]:
    """Dispatch one request line and return its response or error envelope.

    Never raises: the serve loop must survive any single bad request. Full tracebacks
    go to stderr so the persistent transport stays as debuggable as the one-shot one.
    """
    try:
        request = json.loads(line)
    except ValueError as exc:
        return _serve_error("invalid_json", str(exc))
    if not isinstance(request, dict):
        return _serve_error(
            "invalid_request",
            f"expected a JSON object request, got {type(request).__name__}",
        )
    _FSM_CACHE.prepare(request.get("mode"))
    try:
        return handle_request(request)
    except SystemExit as exc:
        # Trust-boundary rejection: the same condition the one-shot transport reports
        # through a non-zero exit code.
        _FSM_CACHE.invalidate()
        return _serve_error("SystemExit", str(exc))
    except Exception as exc:
        # One bad request must not end the loop; KeyboardInterrupt and friends still do.
        _FSM_CACHE.invalidate()
        traceback.print_exc(file=sys.stderr)
        return _serve_error(type(exc).__name__, str(exc))


def serve(stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> int:
    """Serve newline-delimited JSON bridge requests until stdin reaches EOF.

    Exactly one single-line JSON response is written per non-empty request line and
    flushed immediately. ``json.dumps`` escapes embedded newlines, so a response can
    never span lines. When the process stdout is used as the response stream, the
    module-level ``sys.stdout`` is rebound to stderr for the duration of the loop: an
    accidental ``print()`` anywhere in the import graph then lands in the diagnostics
    stream instead of splitting a response.

    Args:
        stdin: Request stream; defaults to the process stdin.
        stdout: Response stream; defaults to the process stdout.

    Returns:
        ``0`` on a clean EOF shutdown.
    """
    source = sys.stdin if stdin is None else stdin
    sink = sys.stdout if stdout is None else stdout
    guard_stdout = stdout is None
    saved_stdout = sys.stdout
    _FSM_CACHE.arm()
    if guard_stdout:
        sys.stdout = sys.stderr
    try:
        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue
            response = _serve_one(line)
            try:
                payload = json.dumps(response)
            except (TypeError, ValueError) as exc:
                payload = json.dumps(_serve_error("unserializable_response", str(exc)))
            sink.write(payload)
            sink.write("\n")
            sink.flush()
    finally:
        if guard_stdout:
            sys.stdout = saved_stdout
    return 0


def main(argv: Sequence[str] = ()) -> int:
    """Run the bridge: one-shot by default, persistent under ``--serve``."""
    args = list(argv)
    if args == ["--serve"]:
        return serve()
    if args:
        raise SystemExit(f"m3a_adapter: unexpected arguments {args}")
    request = json.loads(sys.stdin.read())
    response = handle_request(request)
    sys.stdout.write(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

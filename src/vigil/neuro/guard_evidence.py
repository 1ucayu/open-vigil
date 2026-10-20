"""Per-transition guard evidence view for contract-first guard generation.

A :class:`GuardEvidence` is the deterministic, LLM-free evidence bundle for one FSM
transition that the later typed-``GuardContract`` synthesis pass consumes. It joins:

- the source/target :class:`~vigil.neuro.guard_registry.WidgetRegistry`,
- the proposed canonical action and its resolved stable target alias,
- sibling outgoing transitions from the same source state,
- transition provenance,
- a lightweight deterministic source-to-target diff summary,
- Hoare-style source/target observation evidence (XML/screenshot paths, compact tree,
  and LLM-derived alt text), and
- static-prior hints (activity label, permissions, string resources).

This module only *reads* the existing FSM and raw trace screens. It does not add edges,
modify state identity, change transition evidence, compile DSL, validate admission, or call
an LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from vigil.models.fsm import is_value_payload_action
from vigil.models.guard import RequirednessEvidenceRef
from vigil.neuro.guard_registry import (
    WidgetRegistry,
    build_widget_registry,
    build_widget_registry_from_screen_ids,
)
from vigil.symbolic.screen_context import runtime_elements_from_observation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vigil.models.fsm import AbstractState, AppFSM, Transition
    from vigil.neuro.app_prior import AppPrior


class ScreenEvidence(BaseModel):
    """Observation evidence for one side of a transition.

    Paths are provenance/debug metadata. The prompt also carries textual content
    (compact tree and XML text) because an ordinary LLM API cannot open local paths.
    Screenshots are attached separately by the LLM call path when the file exists.
    """

    state_id: str
    screen_id: str = ""
    activity_name: str = ""
    package_name: str = ""
    screenshot_path: str = ""
    screenshot_screen_id: str = ""
    screenshot_provenance: (
        Literal[
            "selected_observation",
            "same_abstract_state_observation",
        ]
        | None
    ) = None
    xml_tree_path: str = ""
    compact_tree_text: str = ""
    alt_text: str = ""
    page_function: str = ""
    display_name: str = ""


class TransitionEffectMetadata(BaseModel):
    """Evidence-grounded semantic effect annotation for one transition.

    This is an offline annotation consumed by guard generation, never transition proof or
    a runtime verdict.  Role names and descriptions are evidence-derived; the fixed
    ``effect_kind`` vocabulary expresses only framework-level target/payload/effect scope.
    """

    effect_id: str
    effect_kind: Literal["target", "payload"]
    role: str = ""
    description: str = ""


class ResolvedActionSemanticSurface(BaseModel):
    """Live action target plus its nearest stable readable ancestor surface."""

    target_element_id: str
    target_resource_id: str = ""
    surface_element_id: str
    surface_resource_id: str
    descendant_texts: list[str] = Field(default_factory=list)
    subtree_elements: list[dict[str, Any]] = Field(default_factory=list)


class GuardEvidence(BaseModel):
    """Deterministic evidence bundle for one transition's guard synthesis."""

    transition_index: int
    source_state_id: str
    target_state_id: str
    action: dict[str, Any]
    source_page_function: str = ""
    target_page_function: str = ""
    source_screen_ids: list[str] = Field(default_factory=list)
    target_screen_ids: list[str] = Field(default_factory=list)
    source_observations: list[dict[str, Any]] = Field(default_factory=list)
    source_screen: ScreenEvidence = Field(default_factory=lambda: ScreenEvidence(state_id=""))
    target_screen: ScreenEvidence = Field(default_factory=lambda: ScreenEvidence(state_id=""))
    source_registry: WidgetRegistry
    sibling_actions: list[dict[str, Any]] = Field(default_factory=list)
    action_target_alias: str | None = None
    action_target_alias_reason: str = ""
    action_semantic_surface_alias: str | None = None
    action_semantic_surface: ResolvedActionSemanticSurface | None = None
    runtime_intent_feedback: dict[str, Any] = Field(default_factory=dict)
    # Vigil-owned frozen intent-slot fingerprints from the admitted planner proposal's
    # binding snapshot. A new intent-dependent guard clause admitted during Contract
    # enrichment must match the frozen fingerprint (name/type/enum/description) for its slot.
    frozen_slot_fingerprints: dict[str, str] = Field(default_factory=dict)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    static_prior_hints: list[str] = Field(default_factory=list)
    diff_summary: str = ""
    effect_metadata: list[TransitionEffectMetadata] = Field(default_factory=list)
    exploration_payload_values: list[str] = Field(default_factory=list)


def evidence_reference_catalog(evidence: GuardEvidence) -> list[RequirednessEvidenceRef]:
    """Build the exact evidence-reference catalog for synthesis and admission.

    Catalog order is deterministic by evidence family. Duplicate references are retained
    only at their first position. ``input_text`` is deliberately absent from
    ``known_action`` because an offline input value is an exploration probe; the executable
    payload channel is cited through ``effect_metadata/action_value_payload`` instead.

    Args:
        evidence: Concrete transition evidence packet.

    Returns:
        Stable, de-duplicated references that contracts may cite verbatim.
    """
    candidates: list[RequirednessEvidenceRef] = []
    for identifier in ("type", "target_text", "target_content_desc"):
        if evidence.action.get(identifier) not in (None, ""):
            candidates.append(RequirednessEvidenceRef(source="known_action", identifier=identifier))
    candidates.extend(
        RequirednessEvidenceRef(source="source_widget", identifier=alias)
        for alias in sorted(evidence.source_registry.entries)
        if alias
    )
    if evidence.diff_summary:
        candidates.append(
            RequirednessEvidenceRef(
                source="source_target_diff",
                identifier="source_to_target",
            )
        )
    candidates.extend(
        RequirednessEvidenceRef(source="effect_metadata", identifier=effect_id)
        for effect_id in sorted(
            effect.effect_id for effect in evidence.effect_metadata if effect.effect_id
        )
    )
    candidates.extend(
        RequirednessEvidenceRef(source="sibling_action", identifier=str(index))
        for index in range(len(evidence.sibling_actions))
    )
    candidates.extend(
        RequirednessEvidenceRef(source="trace", identifier=str(trace_index))
        for item in evidence.provenance
        if (trace_index := item.get("trace_step_index")) is not None
    )

    catalog: list[RequirednessEvidenceRef] = []
    seen: set[tuple[str, str]] = set()
    for ref in candidates:
        key = (ref.source, ref.identifier)
        if key in seen:
            continue
        seen.add(key)
        catalog.append(ref)
    return catalog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _selector_resource_id(action: dict[str, Any]) -> str:
    selector = action.get("target_selector")
    if isinstance(selector, dict):
        return str(selector.get("resource_id") or "")
    return ""


def _descendant_elements(
    root: dict[str, Any],
    elements_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    descendants: list[dict[str, Any]] = []
    seen: set[str] = set()
    stack = list(root.get("children") or [])
    while stack:
        child_ref = stack.pop(0)
        child = child_ref if isinstance(child_ref, dict) else elements_by_id.get(str(child_ref))
        if not isinstance(child, dict):
            continue
        child_id = str(child.get("element_id") or "")
        if child_id and child_id in seen:
            continue
        if child_id:
            seen.add(child_id)
        descendants.append(child)
        stack.extend(child.get("children") or [])
    return descendants


def resolve_action_semantic_surface(
    observation: Any,
    bound_action: dict[str, Any],
) -> ResolvedActionSemanticSurface | None:
    """Resolve a component target and its nearest stable readable ancestor.

    Resolution uses only concrete hierarchy and stable resource identity.  It never
    interprets resource-id tokens or visible labels.  The ancestor surface is useful
    when the clicked leaf is unlabeled but a containing item exposes sibling text.
    """
    elements = runtime_elements_from_observation(observation)
    elements_by_id = {
        str(element.get("element_id")): element for element in elements if element.get("element_id")
    }
    target: dict[str, Any] | None = None
    target_id = str(bound_action.get("target") or "")
    if target_id:
        target = elements_by_id.get(target_id)
    if target is None:
        resource_id = str(
            bound_action.get("target_resource_id")
            or bound_action.get("resource_id")
            or _selector_resource_id(bound_action)
            or ""
        )
        matches = [
            element
            for element in elements
            if resource_id and str(element.get("resource_id") or "") == resource_id
        ]
        if len(matches) == 1:
            target = matches[0]
    if target is None:
        return None

    ancestors: list[dict[str, Any]] = []
    parent_id = str(target.get("parent_id") or "")
    seen_parents: set[str] = set()
    while parent_id and parent_id not in seen_parents:
        seen_parents.add(parent_id)
        parent = elements_by_id.get(parent_id)
        if parent is None:
            break
        ancestors.append(parent)
        parent_id = str(parent.get("parent_id") or "")

    surface: dict[str, Any] | None = None
    surface_descendants: list[dict[str, Any]] = []
    for ancestor in ancestors:
        resource_id = str(ancestor.get("resource_id") or "").strip()
        descendants = _descendant_elements(ancestor, elements_by_id)
        readable = [
            str(item.get("text") or item.get("content_description") or "").strip()
            for item in descendants
        ]
        if resource_id and any(readable):
            surface = ancestor
            surface_descendants = descendants
            break

    if surface is None and str(target.get("resource_id") or "").strip():
        surface = target
        surface_descendants = _descendant_elements(target, elements_by_id)
    if surface is None:
        return None

    descendant_texts: list[str] = []
    for item in surface_descendants:
        value = str(item.get("text") or item.get("content_description") or "").strip()
        if value and value not in descendant_texts:
            descendant_texts.append(value)
    subtree = [surface, *surface_descendants]
    return ResolvedActionSemanticSurface(
        target_element_id=str(target.get("element_id") or ""),
        target_resource_id=str(target.get("resource_id") or ""),
        surface_element_id=str(surface.get("element_id") or ""),
        surface_resource_id=str(surface.get("resource_id") or ""),
        descendant_texts=descendant_texts,
        subtree_elements=[dict(element) for element in subtree],
    )


def _resolve_target_alias(
    action: dict[str, Any], source_registry: WidgetRegistry
) -> tuple[str | None, str]:
    """Resolve the proposed action target to a stable source-registry alias.

    Returns ``(alias, reason)``. ``alias`` is ``None`` when the target cannot be
    resolved unambiguously; ``reason`` is an observable explanation (e.g. a text match
    that hit zero or multiple entries) for evidence/report metadata.
    """
    # 1. capture-local element handle.
    target = str(action.get("target") or "")
    if target and target in source_registry.element_id_to_alias:
        return source_registry.element_id_to_alias[target], "matched:element_id"

    # 2. resource id from the action or its selector.
    for rid in (
        str(action.get("target_resource_id") or ""),
        str(action.get("resource_id") or ""),
        _selector_resource_id(action),
    ):
        if rid and rid in source_registry.resource_id_to_alias:
            return source_registry.resource_id_to_alias[rid], "matched:resource_id"

    # 3. text match against registry entry text — only when exactly one entry matches.
    if not is_value_payload_action(action):
        selector = action.get("target_selector")
        selector_text = selector.get("text") if isinstance(selector, dict) else None
        target_text = str(action.get("target_text") or selector_text or "").strip()
        if target_text:
            matches = [
                entry.alias
                for entry in source_registry.entries.values()
                if entry.text.strip() == target_text
            ]
            if len(matches) == 1:
                return matches[0], "matched:text"
            if len(matches) > 1:
                return None, (f"ambiguous:text={target_text!r} matched {len(matches)} entries")
            return None, f"unresolved:no_text_match={target_text!r}"

    return None, "unresolved:no_target_signal"


def _evidence_action(transition: Transition) -> dict[str, Any]:
    """Return the trace-level action sample for evidence-only consumers."""
    for entry in transition.provenance:
        if entry.raw_action:
            return entry.raw_action
    return transition.action


def _sibling_actions(fsm: AppFSM, transition: Transition) -> list[dict[str, Any]]:
    return [
        _evidence_action(t)
        for t in fsm.transitions
        if t.source == transition.source and t is not transition
    ]


def _provenance_screen_ids(transition: Transition, side: Literal["source", "target"]) -> list[str]:
    """Ordered, de-duplicated provenance screen ids for one side of a transition.

    ``side`` is ``"source"`` or ``"target"``. These screens are the exact captures
    before/after the action, so they deterministically contain the capture-local action
    target handle even when a merged/quotiented state spans several raw screens.
    """
    ids: list[str] = []
    for entry in transition.provenance:
        sid = entry.source_screen_id if side == "source" else entry.target_screen_id
        if sid and sid not in ids:
            ids.append(sid)
    return ids


def _first_observation(
    candidate_ids: list[str],
    raw_screens: dict[str, Any],
) -> dict[str, Any] | None:
    for sid in candidate_ids:
        obs = raw_screens.get(sid)
        if isinstance(obs, BaseModel):
            obs = obs.model_dump()
        if isinstance(obs, dict):
            return obs
    return None


def _observations(
    candidate_ids: list[str],
    raw_screens: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve every cited runtime observation in provenance order."""
    resolved: list[dict[str, Any]] = []
    for sid in candidate_ids:
        observation = raw_screens.get(sid)
        if isinstance(observation, BaseModel):
            observation = observation.model_dump()
        if isinstance(observation, dict):
            resolved.append(observation)
    return resolved


def _first_existing_screenshot(
    candidate_ids: list[str],
    raw_screens: dict[str, Any],
) -> tuple[str, str] | None:
    """Return the first on-disk screenshot in deterministic observation order."""
    for screen_id in candidate_ids:
        observation = raw_screens.get(screen_id)
        if isinstance(observation, BaseModel):
            observation = observation.model_dump()
        if not isinstance(observation, dict):
            continue
        raw_path = str(observation.get("screenshot_path") or "")
        if raw_path:
            path = Path(raw_path)
            if path.is_file():
                return screen_id, raw_path
    return None


def exploration_payload_values(transition: Transition) -> list[str]:
    """Return concrete input payloads introduced by offline exploration traces.

    The provenance source establishes that these are harness observations; their
    lexical value is never used to recognize them. Runtime-fallback actions are agent
    actions and therefore are not classified as exploration probes here.
    """
    values: list[str] = []
    for entry in transition.provenance:
        if entry.evidence_source != "observed":
            continue
        action = entry.raw_action
        if not is_value_payload_action(action):
            continue
        for field in ("input_text", "text", "value"):
            value = action.get(field)
            if isinstance(value, str) and value and value not in values:
                values.append(value)
                break
    return values


def _screen_evidence(
    state: AbstractState,
    state_id: str,
    screen_ids: list[str],
    screenshot_candidate_ids: list[str],
    raw_screens: dict[str, Any],
) -> ScreenEvidence:
    """Build Hoare evidence for one transition side.

    Text, XML, activity, and package facts always come from the first selected
    observation in ``screen_ids``.  If that observation has no screenshot, visual
    grounding may use the first existing screenshot among the ordered observations of
    the same abstract state.  The distinct screen id and provenance marker prevent that
    representative image from being presented as a live/preferred capture.
    """
    annotations = state.annotations
    obs = _first_observation(screen_ids, raw_screens)
    if obs is None:
        return ScreenEvidence(
            state_id=state_id,
            alt_text=annotations.alt_text,
            page_function=annotations.page_function,
            display_name=annotations.display_name,
        )

    primary_screen_id = str(obs.get("screen_id") or screen_ids[0])
    screenshot_path = str(obs.get("screenshot_path") or "")
    screenshot_screen_id = primary_screen_id if screenshot_path else ""
    screenshot_provenance: (
        Literal[
            "selected_observation",
            "same_abstract_state_observation",
        ]
        | None
    ) = "selected_observation" if screenshot_path else None
    if not screenshot_path:
        representative = _first_existing_screenshot(screenshot_candidate_ids, raw_screens)
        if representative is not None:
            screenshot_screen_id, screenshot_path = representative
            screenshot_provenance = "same_abstract_state_observation"

    xml_tree_path = str(obs.get("xml_tree_path") or "")
    return ScreenEvidence(
        state_id=state_id,
        screen_id=primary_screen_id,
        activity_name=str(obs.get("activity_name") or ""),
        package_name=str(obs.get("package_name") or ""),
        screenshot_path=screenshot_path,
        screenshot_screen_id=screenshot_screen_id,
        screenshot_provenance=screenshot_provenance,
        xml_tree_path=xml_tree_path,
        compact_tree_text=str(obs.get("compact_tree_text") or "").strip(),
        alt_text=annotations.alt_text,
        page_function=annotations.page_function,
        display_name=annotations.display_name,
    )


def _side_registry(
    state: AbstractState,
    provenance_ids: list[str],
    raw_screens: dict[str, Any],
    app_prior: AppPrior | None,
) -> WidgetRegistry:
    """Build a side registry, preferring provenance screens over state evidence."""
    registry = build_widget_registry_from_screen_ids(
        state.state_id,
        provenance_ids,
        raw_screens,
        app_prior,
    )
    if registry is not None:
        return registry
    # Fall back to the state's representative raw screen.
    return build_widget_registry(state, raw_screens, app_prior)


def _diff_summary(
    source_registry: WidgetRegistry,
    target_registry: WidgetRegistry,
) -> str:
    """Deterministic, lightweight source-to-target diff keyed by resource id."""

    def by_resource(reg: WidgetRegistry) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for entry in reg.entries.values():
            if entry.resource_id:
                out.setdefault(entry.resource_id, entry)
        return out

    src = by_resource(source_registry)
    tgt = by_resource(target_registry)

    parts: list[str] = []

    # Resource ids gained / lost between source and target.
    added = sorted(set(tgt) - set(src))
    removed = sorted(set(src) - set(tgt))
    for rid in added:
        parts.append(f"+{rid}")
    for rid in removed:
        parts.append(f"-{rid}")

    # Changed text / checked / enabled facts for shared resource ids.
    for rid in sorted(set(src) & set(tgt)):
        s_entry = src[rid]
        t_entry = tgt[rid]
        if s_entry.text != t_entry.text:
            parts.append(f"{rid}.text:{s_entry.text!r}->{t_entry.text!r}")
        for prop in ("is_checked", "is_enabled"):
            if (
                prop in s_entry.observed_values
                and prop in t_entry.observed_values
                and s_entry.observed_values[prop] != t_entry.observed_values[prop]
            ):
                parts.append(
                    f"{rid}.{prop}:{s_entry.observed_values[prop]!r}"
                    f"->{t_entry.observed_values[prop]!r}"
                )

    return "; ".join(parts)


def _transition_effect_metadata(
    action: dict[str, Any],
    action_target_alias: str | None,
    action_semantic_surface_alias: str | None,
    source_registry: WidgetRegistry,
) -> list[TransitionEffectMetadata]:
    """Build only target/payload channels directly established by trace evidence."""
    effects: list[TransitionEffectMetadata] = []
    if action_target_alias is not None:
        entry = source_registry.entries[action_target_alias]
        effects.append(
            TransitionEffectMetadata(
                effect_id="resolved_action_target",
                effect_kind="target",
                role=action_target_alias,
                description=(
                    f"Proposed action resolves to a source widget with role {entry.role.value}"
                ),
            )
        )
    if (
        action_semantic_surface_alias is not None
        and action_semantic_surface_alias != action_target_alias
    ):
        effects.append(
            TransitionEffectMetadata(
                effect_id="resolved_action_semantic_surface",
                effect_kind="target",
                role=action_semantic_surface_alias,
                description=(
                    "The bound action target is contained by this stable source-side "
                    "semantic surface"
                ),
            )
        )
    if is_value_payload_action(action):
        effects.append(
            TransitionEffectMetadata(
                effect_id="action_value_payload",
                effect_kind="payload",
                role="input_text",
                description=(
                    "Proposed action carries a runtime value payload; the exploration "
                    "sample is not retained as intent or a guard literal"
                ),
            )
        )
    return effects


def static_prior_hints(
    source: AbstractState | None,
    app_prior: AppPrior | None,
    registry: WidgetRegistry | None = None,
) -> list[str]:
    """List source-matched static priors, without treating them as runtime evidence."""
    if app_prior is None:
        return []

    hints: list[str] = []
    activity = None
    if source is not None:
        activity = source.android_context.activity_name
    if activity:
        for info in app_prior.activities:
            if info.name == activity:
                if info.label:
                    hints.append(f"activity_role:{info.label}")
                break

    for perm in app_prior.permissions:
        hints.append(f"perm:{perm.rsplit('.', 1)[-1]}")

    visible_values: set[str] = set()
    resource_ids: set[str] = set()
    resource_names: set[str] = set()
    if registry is not None:
        for entry in registry.entries.values():
            if entry.text:
                visible_values.add(entry.text)
            if entry.content_description:
                visible_values.add(entry.content_description)
            if entry.resource_id:
                resource_ids.add(entry.resource_id)
                resource_names.add(entry.resource_id.rsplit("/", 1)[-1])
                resource_names.add(entry.resource_id.rsplit(":id/", 1)[-1])

    for name in sorted(app_prior.string_arrays):
        values = app_prior.string_arrays.get(name) or []
        matched = [str(value) for value in values if str(value) in visible_values]
        if matched:
            rendered_values = ", ".join(repr(value) for value in matched)
            hints.append(f"string_array:{name} visible_values=[{rendered_values}]")

    for name, value in sorted(app_prior.string_constants.items()):
        if value in visible_values:
            hints.append(f"string:{name}={value!r}")

    for decl in app_prior.widget_declarations:
        if decl.widget_id in resource_ids or decl.widget_id in resource_names:
            hints.append(
                f"layout_widget:{decl.layout_file} id={decl.widget_id!r} "
                f"class={decl.widget_class!r}"
            )

    return hints


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_guard_evidence_for_transition(
    fsm: AppFSM,
    transition: Transition,
    transition_index: int,
    raw_screens: dict[str, Any],
    app_prior: AppPrior | None = None,
    *,
    preferred_source_screen_ids: list[str] | None = None,
    preferred_target_screen_ids: list[str] | None = None,
    evidence_action_override: dict[str, Any] | None = None,
    runtime_intent_feedback: dict[str, Any] | None = None,
    frozen_slot_fingerprints: dict[str, str] | None = None,
) -> GuardEvidence:
    """Build the :class:`GuardEvidence` view for one transition in a valid FSM."""
    source = fsm.states[transition.source]
    target = fsm.states[transition.target]
    provenance_source_ids = _provenance_screen_ids(transition, "source")
    provenance_target_ids = _provenance_screen_ids(transition, "target")

    def ordered_ids(preferred: list[str] | None, existing: list[str]) -> list[str]:
        result: list[str] = []
        for screen_id in [*(preferred or []), *existing]:
            if screen_id and screen_id not in result:
                result.append(screen_id)
        return result

    source_screen_ids = ordered_ids(
        preferred_source_screen_ids,
        provenance_source_ids or list(source.evidence.raw_screen_ids),
    )
    target_screen_ids = ordered_ids(
        preferred_target_screen_ids,
        provenance_target_ids or list(target.evidence.raw_screen_ids),
    )
    source_screenshot_ids = ordered_ids(
        preferred_source_screen_ids,
        [*provenance_source_ids, *source.evidence.raw_screen_ids],
    )
    target_screenshot_ids = ordered_ids(
        preferred_target_screen_ids,
        [*provenance_target_ids, *target.evidence.raw_screen_ids],
    )

    source_registry = _side_registry(
        source,
        source_screen_ids,
        raw_screens,
        app_prior,
    )
    target_registry = _side_registry(
        target,
        target_screen_ids,
        raw_screens,
        app_prior,
    )

    evidence_action = evidence_action_override or _evidence_action(transition)
    action_target_alias, alias_reason = _resolve_target_alias(evidence_action, source_registry)
    source_observation = _first_observation(source_screen_ids, raw_screens)
    action_semantic_surface = (
        resolve_action_semantic_surface(source_observation, evidence_action)
        if source_observation is not None
        else None
    )
    action_semantic_surface_alias = (
        source_registry.resource_id_to_alias.get(action_semantic_surface.surface_resource_id)
        if action_semantic_surface is not None
        else None
    )

    return GuardEvidence(
        transition_index=transition_index,
        source_state_id=transition.source,
        target_state_id=transition.target,
        action=transition.action,
        source_page_function=source.annotations.page_function,
        target_page_function=target.annotations.page_function,
        source_screen_ids=source_screen_ids,
        target_screen_ids=target_screen_ids,
        source_observations=_observations(source_screen_ids, raw_screens),
        source_screen=_screen_evidence(
            source,
            transition.source,
            source_screen_ids,
            source_screenshot_ids,
            raw_screens,
        ),
        target_screen=_screen_evidence(
            target,
            transition.target,
            target_screen_ids,
            target_screenshot_ids,
            raw_screens,
        ),
        source_registry=source_registry,
        sibling_actions=_sibling_actions(fsm, transition),
        action_target_alias=action_target_alias,
        action_target_alias_reason=alias_reason,
        action_semantic_surface_alias=action_semantic_surface_alias,
        action_semantic_surface=action_semantic_surface,
        runtime_intent_feedback=dict(runtime_intent_feedback or {}),
        frozen_slot_fingerprints=dict(frozen_slot_fingerprints or {}),
        provenance=[p.model_dump() for p in transition.provenance],
        static_prior_hints=static_prior_hints(source, app_prior, source_registry),
        diff_summary=_diff_summary(source_registry, target_registry),
        effect_metadata=_transition_effect_metadata(
            transition.action,
            action_target_alias,
            action_semantic_surface_alias,
            source_registry,
        ),
        exploration_payload_values=exploration_payload_values(transition),
    )


def build_all_guard_evidence(
    fsm: AppFSM,
    raw_screens: dict[str, Any],
    app_prior: AppPrior | None = None,
) -> list[GuardEvidence]:
    """Build :class:`GuardEvidence` for every transition in ``fsm`` (index-ordered)."""
    return [
        build_guard_evidence_for_transition(fsm, transition, index, raw_screens, app_prior)
        for index, transition in enumerate(fsm.transitions)
    ]

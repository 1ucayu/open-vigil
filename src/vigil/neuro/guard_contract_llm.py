"""LLM-backed, contract-first guard generation (Stage 4).

This module asks an LLM to produce a typed transition
:class:`~vigil.models.guard.GuardContract` for a *single, already-known* transition via
**provider structured output** (a strict
:class:`~vigil.models.llm_structured.LlmTransitionGuardResponse` schema), then converts
the parsed object into a
:class:`~vigil.models.guard.LlmGuardContractCandidate`.

Design constraints (CLAUDE.md -> "DSL Guard Generation Direction"; plan):

- The LLM never emits free-form DSL and never raw/fenced JSON — it emits a single
  schema-valid ``LlmTransitionGuardResponse`` object. Compilation/admission to executable
  DSL is a later deterministic step. There is no prompt-only JSON parsing and no
  repair-prompt loop on this path.
- The LLM may not create/modify FSM states, actions, transitions, transition evidence, or runtime
  verdicts. Target-state evidence is background context only; executable guard predicates may
  read only the source screen, proposed action, and frozen intent.
- The LLM schema exposes semantic classification evidence, typed ``slots``, and
  independently activatable executable ``clauses``. Requiredness is computed by admission.
  Non-executable binding ideas are omitted; admission reports unsupported semantics.
- When structured output is unavailable (provider/schema failure, refusal, or validation
  failure), the result is a clearly rejected candidate (``parsed_ok=False``) carrying the
  provider/schema error — never a fabricated success.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
from loguru import logger
from openai import OpenAIError

from vigil.models.guard import (
    GuardContract,
    GuardKind,
    LlmGuardContractCandidate,
    RequirednessEvidenceRef,
)
from vigil.models.llm_structured import LlmTransitionGuardResponse
from vigil.neuro.guard_evidence import evidence_reference_catalog
from vigil.system_prompt import load_system_prompt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vigil.core.llm_client import LlmClient
    from vigil.core.structured import StructuredResult
    from vigil.neuro.guard_evidence import GuardEvidence, ScreenEvidence
    from vigil.neuro.guard_registry import WidgetRegistry
    from vigil.neuro.prompt_redaction import PromptRedactor


DEFAULT_GUARD_PROMPT = "transition_guard_generation.spec"
GUARD_SCHEMA_NAME = "LlmTransitionGuardResponse"

_LLM_READABLE_PROPS: frozenset[str] = frozenset(
    {
        "text",
        "content_description",
        "value",
        "is_checked",
        "is_selected",
        "children_count",
    }
)


def registry_lines(registry: WidgetRegistry) -> list[str]:
    lines: list[str] = []
    for alias, entry in registry.entries.items():
        if not entry.resource_id:
            continue
        props = ", ".join(prop for prop in entry.readable_props if prop in _LLM_READABLE_PROPS)
        parts = [f"- alias={alias}", f"resource_id={entry.resource_id}"]
        if entry.text:
            parts.append(f"text={entry.text!r}")
        if entry.content_description:
            parts.append(f"content_desc={entry.content_description!r}")
        if entry.descendant_texts:
            parts.append(f"visible_descendant_texts={entry.descendant_texts!r}")
        parts.append(f"role={entry.role.value}")
        if props:
            parts.append(f"readable=[{props}]")
        lines.append(" ".join(parts))
    return lines


def _sibling_lines(siblings: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for action in siblings:
        atype = str(action.get("type") or "")
        text = str(action.get("target_text") or "").strip()
        suffix = f" target_text={text!r}" if text else ""
        lines.append(f"- {atype}{suffix}")
    return lines


def _effect_lines(evidence: GuardEvidence) -> list[str]:
    """Render typed effect metadata and its admissible evidence identifier."""
    return [
        (
            f"- id={effect.effect_id!r} kind={effect.effect_kind!r} role={effect.role!r} "
            f"description={effect.description!r}"
        )
        for effect in evidence.effect_metadata
    ]


def _evidence_ref_line(ref: RequirednessEvidenceRef) -> str:
    """Render one evidence reference in the structured-output schema shape."""
    return f"- {json.dumps(ref.model_dump(mode='json'), ensure_ascii=False)}"


def _protected_protocol_fragments(
    evidence: GuardEvidence,
    existing_contract: GuardContract | None,
) -> list[str]:
    """Return rendered prompt fragments whose identifiers must remain byte-exact."""
    catalog = evidence_reference_catalog(evidence)
    fragments = [json.dumps(ref.model_dump(mode="json"), ensure_ascii=False) for ref in catalog]
    for alias in evidence.source_registry.entries:
        fragments.extend((f"alias={alias}", repr(alias)))
    if existing_contract is not None:
        slot_names = [slot.name for slot in existing_contract.slots]
        fragments.append(repr(slot_names))
        fragments.extend(f"$intent.{name}" for name in slot_names)
    return fragments


def _retry_diagnostics_slot_names(retry_diagnostics: dict[str, Any]) -> list[str]:
    """Canonical slot names named by prior-attempt coherence diagnostics (order-preserving)."""
    names: list[str] = []
    for conflict in retry_diagnostics.get("coherence_conflicts", []):
        candidate_names = [conflict.get("slot_name")]
        candidate_names.extend(occ.get("slot_name") for occ in conflict.get("occurrences", []))
        names.extend(name for name in candidate_names if name)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _retry_diagnostics_section(retry_diagnostics: dict[str, Any]) -> str:
    """Render the compact, deterministic prior-attempt admission diagnostics section.

    The payload is the authoritative deterministic-admission report from the previous
    source-state attempt (per-transition admission failures plus same-name slot-fingerprint
    coherence conflicts carrying exact name/type/enum-domain/fingerprint per transition).
    """
    payload = {
        "generation_failures": retry_diagnostics.get("generation_failures", []),
        "admission_failures": retry_diagnostics.get("admission_failures", []),
        "coherence_conflicts": retry_diagnostics.get("coherence_conflicts", []),
    }
    return (
        "[Prior attempt admission diagnostics]\n"
        "Purpose: Authoritative deterministic-admission reports from the previous attempt at "
        "this source state. Reconsider this transition's contract from the evidence; do not "
        "normalize, copy, or force a schema. When a canonical slot name is reused across sibling "
        "outgoing transitions, its slot_type and enum domain must be consistent.\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _action_prompt_view(action: dict[str, Any]) -> dict[str, Any]:
    """Return semantic action evidence without capture-local/selector identity fields."""
    return {
        key: action[key]
        for key in ("type", "target_text", "target_content_desc", "input_text")
        if action.get(key) not in (None, "")
    }


def _fenced(label: str, value: str) -> str:
    if not value.strip():
        return f"{label}:\n(none)"
    return f"{label}:\n```text\n{value}\n```"


def _screen_section(title: str, screen: ScreenEvidence, *, source_readable: bool) -> str:
    if screen.screenshot_provenance == "same_abstract_state_observation":
        screenshot_status = (
            "available from another observation of the same abstract state; visual context "
            "only, not the selected/live observation"
        )
    elif screen.screenshot_path:
        screenshot_status = "available from the selected observation"
    else:
        screenshot_status = "not available"
    purpose = (
        "This is source observation o_t: the ONLY UI state that transition guard Gamma may read."
        if source_readable
        else (
            "This is successor observation o_{t+1}: background evidence for classifying "
            "the transition. "
            "Do NOT reference target-only elements in guard predicates."
        )
    )

    parts = [
        f"[{title}]",
        f"Purpose: {purpose}",
        f"- activity: {screen.activity_name or '(none)'}",
        f"- display_name: {screen.display_name!r}",
        f"- page_function: {screen.page_function!r}",
        f"- screenshot: {screenshot_status}; not attached unless explicit image mode is enabled",
        _fenced(
            "Visual Caption Cache (screenshot-only perception hint, not admission proof)",
            screen.alt_text,
        ),
        _fenced("Compact accessibility/XML tree summary", screen.compact_tree_text),
    ]
    return "\n".join(parts)


def _existing_image_path(raw_path: str) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_file():
        return path
    return None


def guard_image_paths(evidence: GuardEvidence) -> tuple[list[Path], list[str]]:
    """Return existing source/target screenshots to attach to the LLM request."""
    images: list[Path] = []
    labels: list[str] = []
    seen: set[Path] = set()
    for side, screen, label in (
        (
            "SOURCE",
            evidence.source_screen,
            "SOURCE screenshot: observation o_t. Guard predicates may read this UI.",
        ),
        (
            "TARGET",
            evidence.target_screen,
            "TARGET screenshot: successor observation o_{t+1}. Background evidence only for Gamma.",
        ),
    ):
        path = _existing_image_path(screen.screenshot_path)
        if path is None:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        images.append(path)
        screenshot_screen_id = screen.screenshot_screen_id or screen.screen_id or side.lower()
        if screen.screenshot_provenance == "same_abstract_state_observation":
            labels.append(
                f"{label} Representative visual capture from the same abstract state; "
                "it is not the selected/live observation. "
                f"state={screen.state_id} selected_screen={screen.screen_id or side.lower()} "
                f"screenshot_screen={screenshot_screen_id}"
            )
        else:
            labels.append(f"{label} state={screen.state_id} screen={screenshot_screen_id}")
    return images, labels


def build_guard_user_prompt(
    evidence: GuardEvidence,
    *,
    redactor: PromptRedactor | None = None,
    existing_contract: GuardContract | None = None,
    retry_diagnostics: dict[str, Any] | None = None,
) -> str:
    """Build the user prompt for one transition guard contract.

    When ``redactor`` is supplied, application identifiers (package names, app slugs,
    raw screen ids, local paths, evaluator labels) are masked in the assembled prompt while
    usable registry aliases, permissions, and action properties are preserved. When
    ``existing_contract`` is supplied (additive Contract enrichment), its already-admitted
    clauses are shown so the model proposes only *additional* or *specialized* clauses
    rather than replacements. When ``retry_diagnostics`` is supplied (a bounded-retry
    attempt), the prior attempt's deterministic admission report is appended so the model can
    reconsider this transition's candidate from the evidence.
    """
    reg_lines = registry_lines(evidence.source_registry)
    sib_lines = _sibling_lines(evidence.sibling_actions)
    effect_lines = _effect_lines(evidence)
    hints = evidence.static_prior_hints

    sections: list[str] = []
    sections.append(
        "/* Transition Guard Evidence */\n"
        "Generate a minimal transition guard contract Gamma for this already-known transition.\n"
        "Gamma(s_t, sigma_t) may reference only source observation o_t, proposed action "
        "properties, and frozen "
        "$intent.* variables."
    )

    sections.append(f"[Transition]\n- source_page_function: {evidence.source_page_function!r}")

    surface_texts = (
        evidence.action_semantic_surface.descendant_texts
        if evidence.action_semantic_surface is not None
        else []
    )
    sections.append(
        "[Known action]\n"
        "Purpose: Fixed transition action.\n"
        f"{json.dumps(_action_prompt_view(evidence.action), ensure_ascii=False, indent=2)}\n"
        f"- resolved source-widget alias: {evidence.action_target_alias!r} "
        f"({evidence.action_target_alias_reason})\n"
        f"- nearest stable ancestor semantic surface: "
        f"{evidence.action_semantic_surface_alias!r}\n"
        "- semantic-surface visible descendant texts: "
        f"{surface_texts!r}"
    )

    sections.append(
        "[Exploration action payload]\n"
        "Any input value carried by this offline exploration action is a harness probe, "
        "not user intent and not a guard literal. For input_binding, bind the proposed "
        "runtime input to a declared $intent.* slot."
    )

    if evidence.runtime_intent_feedback:
        sections.append(
            "[Runtime semantic-gap intent feedback]\n"
            "Purpose: mobile-agent planner evidence for targeted Contract enrichment. "
            "A bound value may identify a typed intent relation, but must be represented "
            "as a $intent.* slot and never copied into a literal predicate.\n"
            f"{json.dumps(evidence.runtime_intent_feedback, ensure_ascii=False, indent=2)}"
        )

    sections.append(
        _screen_section(
            "Source observation evidence: o_t",
            evidence.source_screen,
            source_readable=True,
        )
    )

    sections.append(
        "[Source widget registry]\n"
        "Purpose: The ONLY source element aliases executable predicates may reference.\n"
        + ("\n".join(reg_lines) if reg_lines else "- (none resolved)")
    )

    sections.append(
        _screen_section(
            "Successor observation evidence: o_{t+1} (background only)",
            evidence.target_screen,
            source_readable=False,
        )
    )

    sections.append(
        "[Source-to-target semantic/evidence diff]\n"
        "Purpose: Transition classification context; do not predicate on target-only UI "
        "in Gamma.\n"
        f"- target_page_function: {evidence.target_page_function!r}\n"
        f"- source->target diff: {evidence.diff_summary or '(none)'}"
    )

    sections.append(
        "[Typed transition effect metadata]\n"
        "Purpose: Evidence-grounded target/payload/committed-effect roles for the slot-removal "
        "counterfactual. These annotations are semantic evidence, not transition proof.\n"
        + ("\n".join(effect_lines) if effect_lines else "- (none)")
    )

    evidence_ref_lines = [_evidence_ref_line(ref) for ref in evidence_reference_catalog(evidence)]
    sections.append(
        "[Semantic evidence reference catalog]\n"
        "Purpose: Every semantic_evidence entry must copy "
        "one complete JSON object from this catalog field-for-field. Do not parse, concatenate, "
        "namespace, or invent references.\n"
        + ("\n".join(evidence_ref_lines) if evidence_ref_lines else "- (none)")
    )

    sections.append(
        "[Sibling outgoing actions]\n"
        "Purpose: Distinguish choices, form controls, commits, navigation, and cancel actions.\n"
        + ("\n".join(sib_lines) if sib_lines else "- (none)")
    )

    sections.append(
        "[Global Information / Static APK Priors]\n"
        "Purpose: Hints only; never transition proof, runtime proof, or proof of "
        "guard satisfaction.\n" + ("\n".join(f"- {h}" for h in hints) if hints else "- (none)")
    )

    sections.append(
        "[Output]\n"
        "Emit only the minimal transition guard contract object described in the system prompt: "
        "typed slots plus one independently meaningful predicate per clause."
    )

    if existing_contract is not None:
        admitted_clauses = [
            clause.compiled_expr for clause in existing_contract.clauses if clause.compiled_expr
        ]
        sections.append(
            "[Already-admitted guard contract]\n"
            "Purpose: This transition already carries the admitted clauses below "
            "(additive enrichment). Propose ONLY additional or more specialized clauses "
            "and slots; do not restate or weaken these.\n"
            f"- kind: {existing_contract.kind.value}\n"
            f"- admitted slots: {[slot.name for slot in existing_contract.slots]!r}\n"
            + (
                "\n".join(f"- admitted clause: {expr}" for expr in admitted_clauses)
                if admitted_clauses
                else "- admitted clause: (none)"
            )
        )

    if retry_diagnostics:
        sections.append(_retry_diagnostics_section(retry_diagnostics))

    prompt = "\n\n".join(sections)
    if redactor is not None:
        protected = _protected_protocol_fragments(evidence, existing_contract)
        if retry_diagnostics:
            for name in _retry_diagnostics_slot_names(retry_diagnostics):
                protected.extend((name, f"$intent.{name}"))
        prompt = redactor.redact(prompt, protected_literals=protected)
    return prompt


# ---------------------------------------------------------------------------
# Structured-result conversion
# ---------------------------------------------------------------------------


def failure_reason(result: StructuredResult) -> str:
    """Human-readable reason for a structured result that produced no parsed object."""
    if result.refusal:
        return f"provider refusal: {result.refusal}"
    if result.schema_constraint_mode == "prompt_only_unavailable":
        detail = result.validation_errors[0] if result.validation_errors else ""
        return f"structured output unavailable: {detail}".strip()
    if result.validation_errors:
        return f"schema validation failed: {result.validation_errors[0]}"
    if result.incomplete:
        return f"incomplete response: {result.incomplete_detail or 'truncated'}"
    return "no schema-valid object returned"


def candidate_from_structured_result(result: StructuredResult) -> LlmGuardContractCandidate:
    """Convert the client-validated transition response into a runtime candidate."""
    if result.parsed is not None:
        candidate = cast(LlmTransitionGuardResponse, result.parsed).to_runtime()
        candidate.parsed_ok = True
        return candidate

    return LlmGuardContractCandidate(
        contract=GuardContract(kind=GuardKind.UNKNOWN),
        parsed_ok=False,
        rejection_reason=failure_reason(result),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_llm_guard_candidate(
    evidence: GuardEvidence,
    llm: LlmClient,
    *,
    prompt_name: str = DEFAULT_GUARD_PROMPT,
    use_images: bool = False,
    redactor: PromptRedactor | None = None,
    existing_contract: GuardContract | None = None,
    retry_diagnostics: dict[str, Any] | None = None,
) -> LlmGuardContractCandidate:
    """Generate a guard-contract candidate for one transition via structured output.

    Loads the system prompt, builds the (optionally redacted) user prompt from ``evidence``,
    and calls the provider's schema-constrained structured-output path with the strict
    :class:`LlmTransitionGuardResponse` model. By default this stage does not resend
    screenshots; it consumes the visual caption cache produced by visual grounding. Explicit
    image mode remains available for debugging or low-confidence perception fallback. The
    parsed object is converted directly into a candidate; a structured-output failure yields a
    clearly rejected candidate (``parsed_ok=False``) rather than a fabricated success. No
    prompt-only JSON parsing or repair re-prompts run on this path. When ``existing_contract``
    is supplied the prompt asks for additive-only enrichment.
    """
    system_prompt = load_system_prompt(prompt_name)
    user_prompt = build_guard_user_prompt(
        evidence,
        redactor=redactor,
        existing_contract=existing_contract,
        retry_diagnostics=retry_diagnostics,
    )
    try:
        images, image_labels = guard_image_paths(evidence) if use_images else ([], [])
        if redactor is not None:
            image_labels = [redactor.redact(label) for label in image_labels]
        if images:
            result = llm.generate_structured_with_images(
                system_prompt,
                user_prompt,
                images,
                LlmTransitionGuardResponse,
                GUARD_SCHEMA_NAME,
                image_labels,
            )
        else:
            result = llm.generate_structured(
                system_prompt,
                user_prompt,
                LlmTransitionGuardResponse,
                GUARD_SCHEMA_NAME,
            )
    except (httpx.HTTPError, OpenAIError, OSError, json.JSONDecodeError) as exc:
        logger.warning(
            f"LLM guard generation failed for transition {evidence.transition_index}: {exc}"
        )
        return LlmGuardContractCandidate(
            contract=GuardContract(kind=GuardKind.UNKNOWN),
            parsed_ok=False,
            rejection_reason=f"llm call failed: {exc}",
        )
    return candidate_from_structured_result(result)

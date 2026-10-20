"""LLM-backed, contract-first state-invariant candidate generation (Stage 4).

This asks an LLM to produce a typed
:class:`~vigil.models.invariant_candidate.InvariantCandidatePacket` for a *single,
already-built* arrival state via **provider structured output** (a strict
:class:`~vigil.models.llm_structured.LlmInvariantGuardResponse` schema), then converts the
parsed object into a :class:`LlmInvariantPacketCandidate`. The model is a lean candidate
generator only: it proposes minimal typed state-invariant predicates. Deterministic
compilation, admission, DSL parsing, alias resolution, transition evidence, and runtime
verdicts all remain symbolic downstream.

When structured output is unavailable (provider/schema failure, refusal, or validation
failure), the result is a clearly rejected candidate (``parsed_ok=False``) with the
provider/schema error — never a fabricated success. There is no prompt-only JSON parsing and
no repair-prompt loop on this path.
"""

from __future__ import annotations

from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
from loguru import logger
from openai import OpenAIError
from pydantic import BaseModel, Field

from vigil.models.invariant_candidate import InvariantCandidatePacket
from vigil.models.llm_structured import LlmInvariantGuardResponse
from vigil.neuro.guard_contract_llm import failure_reason, registry_lines
from vigil.system_prompt import load_system_prompt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vigil.core.llm_client import LlmClient
    from vigil.core.structured import StructuredResult
    from vigil.neuro.invariant_evidence import InvariantEvidence
    from vigil.neuro.prompt_redaction import PromptRedactor


DEFAULT_INVARIANT_PROMPT = "invariant_guard_generation.spec"
INVARIANT_SCHEMA_NAME = "LlmInvariantGuardResponse"

# Cap how much per-observation detail enters the prompt so multi-visit states stay bounded.
_MAX_PROMPT_OBSERVATIONS = 4
_MAX_OBSERVATION_ELEMENTS = 25
_MAX_PROMPT_IMAGES = 3


class LlmInvariantPacketCandidate(BaseModel):
    """An LLM-produced candidate packet, before admission.

    ``parsed_ok`` is the authoritative success flag: ``True`` only when the provider returned
    a schema-valid object. When ``False`` the pipeline must treat this as a clear rejection
    and must not synthesize/attach invariants as if generation had succeeded.
    """

    packet: InvariantCandidatePacket = Field(default_factory=InvariantCandidatePacket)
    rejection_reason: str = ""
    parsed_ok: bool = False


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _observation_lines(observation: dict[str, Any]) -> list[str]:
    from vigil.neuro.guard_registry import elements_of

    lines: list[str] = []
    for element in elements_of(observation)[:_MAX_OBSERVATION_ELEMENTS]:
        rid = str(element.get("resource_id") or "")
        if not rid:
            continue
        text = str(element.get("text") or "").strip()
        content_description = str(element.get("content_description") or "").strip()
        facts: list[str] = []
        if text:
            facts.append(f"text={text!r}")
        if content_description:
            facts.append(f"content_description={content_description!r}")
        for prop in ("is_checked", "is_selected"):
            if prop in element:
                facts.append(f"{prop}={bool(element.get(prop))}")
        children = element.get("children")
        if isinstance(children, list) and children:
            facts.append(f"children_count={len(children)}")
        suffix = " " + " ".join(facts) if facts else ""
        lines.append(f"  - {rid}{suffix}")
    return lines


def _observations_section(evidence: InvariantEvidence) -> str:
    parts: list[str] = [
        "[State observations]",
        "Purpose: ONLY evidence for state-invariant candidates. Repeated visits are stronger.",
        f"- observation_count: {evidence.observation_count}",
    ]
    for index, observation in enumerate(evidence.observations[:_MAX_PROMPT_OBSERVATIONS]):
        screen_id = str(observation.get("screen_id") or f"obs_{index}")
        parts.append(f"- observation {index} (screen_id={screen_id}):")
        parts.extend(_observation_lines(observation))
    return "\n".join(parts)


def _visual_caption_section(evidence: InvariantEvidence) -> str:
    caption = evidence.visual_alt_text.strip()
    if not caption:
        caption = "(none)"
    return (
        "[Visual Caption Cache]\n"
        "Purpose: screenshot-only perception hint from visual grounding. Use it to "
        "interpret icon meanings, visual grouping, overlays, selected/disabled visual "
        "state, and other facts missing from XML. It is not symbolic admission proof; "
        "executable invariant predicates must still reference arrival-registry aliases; "
        "deterministic admission lowers them to runtime resource ids.\n"
        f"```text\n{caption}\n```"
    )


def _transition_lines(transitions: Any) -> list[str]:
    lines: list[str] = []
    for summary in transitions:
        action_type = str(summary.action.get("type") or "")
        target_text = str(summary.action.get("target_text") or "").strip()
        suffix = f" target_text={target_text!r}" if target_text else ""
        lines.append(f"- action=[{action_type}{suffix}]")
    return lines


def build_invariant_user_prompt(
    evidence: InvariantEvidence,
    *,
    redactor: PromptRedactor | None = None,
) -> str:
    """Build the per-state user prompt for invariant candidate generation.

    When ``redactor`` is supplied, application identifiers are masked in the assembled
    prompt while usable registry aliases, permissions, and action properties are preserved.
    """
    reg_lines = registry_lines(evidence.arrival_registry)
    sections: list[str] = []
    sections.append(
        "/* Minimal invariant synthesis */\n"
        "Generate the typed invariant candidate packet for this ALREADY-BUILT "
        "state. Topology is fixed: do not invent states, transitions, or actions.\n"
        "Runtime state invariants are evaluated with ScreenContext only — typed predicates may use "
        "read/value/count with the allowed operators over the arrival registry and must NOT use "
        "$intent.*, $bind.*, or action(...).\n"
        "These are post-action/current-state consistency facts, never pre-action safety guards. "
        "Emit only state-defining, non-redundant candidates and avoid UI snapshot enumeration. "
        "Choose user-visible title/status anchors that provide independent facts; add another "
        "candidate only for an independent mutable state fact. Ignore machine/debug/evaluator "
        "markers and benchmark-fixture copy."
    )

    sections.append(
        "[Target state]\n"
        f"- window_name: {evidence.window_name!r}\n"
        f"- container_type: {evidence.container_type!r}\n"
        f"- page_function: {evidence.page_function!r}\n"
        f"- display_name: {evidence.display_name!r}"
    )

    sections.append(_visual_caption_section(evidence))

    sections.append(_observations_section(evidence))

    sections.append(
        "[Arrival-state widget registry]\n"
        "Purpose: the ONLY element aliases executable state-invariant predicates may use.\n"
        + ("\n".join(reg_lines) if reg_lines else "- (none resolved)")
    )

    sections.append(
        "[Incoming transitions]\n"
        "Purpose: context for stable arrival facts; do not emit predecessor-specific facts.\n"
        + ("\n".join(_transition_lines(evidence.incoming)) if evidence.incoming else "- (none)")
    )

    sections.append(
        "[Outgoing transitions]\n"
        "Purpose: context for avoiding ordinary action-affordance enumeration.\n"
        + ("\n".join(_transition_lines(evidence.outgoing)) if evidence.outgoing else "- (none)")
    )

    sections.append(
        "[Global Information / Static APK Priors]\n"
        "Purpose: semantic role/domain hints only; never runtime proof.\n"
        + (
            "\n".join(f"- {hint}" for hint in evidence.static_prior_hints)
            if evidence.static_prior_hints
            else "- (none)"
        )
    )

    sections.append(
        "[Output]\n"
        "Emit the typed invariant packet object from the system prompt "
        "(`candidates` only; no DSL expr strings)."
    )

    prompt = "\n\n".join(sections)
    if redactor is not None:
        prompt = redactor.redact(prompt)
    return prompt


def invariant_image_paths(evidence: InvariantEvidence) -> tuple[list[Path], list[str]]:
    """Return existing observation screenshots to attach to the LLM request."""
    images: list[Path] = []
    labels: list[str] = []
    seen: set[Path] = set()
    for observation in evidence.observations:
        raw_path = str(observation.get("screenshot_path") or "")
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        images.append(path)
        labels.append(
            f"Observation screenshot of arrival state {evidence.target_state_id} "
            f"(screen={observation.get('screen_id') or 'obs'})"
        )
        if len(images) >= _MAX_PROMPT_IMAGES:
            break
    return images, labels


# ---------------------------------------------------------------------------
# Structured-result conversion
# ---------------------------------------------------------------------------


def candidate_from_structured_result(result: StructuredResult) -> LlmInvariantPacketCandidate:
    """Convert the client-validated invariant response into a runtime candidate."""
    if result.parsed is not None:
        response = cast(LlmInvariantGuardResponse, result.parsed)
        return LlmInvariantPacketCandidate(packet=response.to_runtime(), parsed_ok=True)

    return LlmInvariantPacketCandidate(
        parsed_ok=False,
        rejection_reason=failure_reason(result),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_llm_invariant_guard_candidate(
    evidence: InvariantEvidence,
    llm: LlmClient,
    *,
    prompt_name: str = DEFAULT_INVARIANT_PROMPT,
    use_images: bool = False,
    redactor: PromptRedactor | None = None,
) -> LlmInvariantPacketCandidate:
    """Generate an invariant/guard candidate packet for one state via structured output."""
    system_prompt = load_system_prompt(prompt_name)
    user_prompt = build_invariant_user_prompt(evidence, redactor=redactor)
    try:
        images, image_labels = invariant_image_paths(evidence) if use_images else ([], [])
        if redactor is not None:
            image_labels = [redactor.redact(label) for label in image_labels]
        if images:
            result = llm.generate_structured_with_images(
                system_prompt,
                user_prompt,
                images,
                LlmInvariantGuardResponse,
                INVARIANT_SCHEMA_NAME,
                image_labels,
            )
        else:
            result = llm.generate_structured(
                system_prompt,
                user_prompt,
                LlmInvariantGuardResponse,
                INVARIANT_SCHEMA_NAME,
            )
    except (httpx.HTTPError, OpenAIError, OSError, JSONDecodeError) as exc:
        logger.warning(
            f"LLM invariant generation failed for state {evidence.target_state_id}: {exc}"
        )
        return LlmInvariantPacketCandidate(
            parsed_ok=False,
            rejection_reason=f"llm call failed: {exc}",
        )
    return candidate_from_structured_result(result)

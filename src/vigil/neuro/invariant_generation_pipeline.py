"""Contract-first state-invariant generation and admission.

For each existing abstract state:

    runtime observations -> typed candidates -> deterministic evidence admission
      -> independently stored StateInvariant entries

The pipeline never generates transition guards, effect hints, edges, state identities,
or transition evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from vigil.models.invariant_candidate import InvariantCandidatePacket
from vigil.neuro.invariant_admission import admit_state_invariant_candidate
from vigil.neuro.invariant_contract_synthesizer import synthesize_invariant_candidates
from vigil.neuro.invariant_evidence import build_all_invariant_evidence
from vigil.neuro.invariant_guard_llm import (
    DEFAULT_INVARIANT_PROMPT,
    generate_llm_invariant_guard_candidate,
)
from vigil.symbolic.dsl_evaluator import DSLEvaluator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vigil.core.llm_client import LlmClient
    from vigil.models.fsm import AppFSM, StateInvariant
    from vigil.neuro.app_prior import AppPrior
    from vigil.neuro.invariant_evidence import InvariantEvidence
    from vigil.neuro.prompt_redaction import PromptRedactor

InvariantSource = Literal["deterministic", "llm"]


def _resolve_candidates(
    evidence: InvariantEvidence,
    source: InvariantSource,
    llm: LlmClient | None,
    prompt: str,
    use_images: bool,
    redactor: PromptRedactor | None,
) -> tuple[InvariantCandidatePacket | None, str, str]:
    if source == "deterministic":
        return synthesize_invariant_candidates(evidence), "deterministic", ""

    candidate = generate_llm_invariant_guard_candidate(
        evidence,
        cast("LlmClient", llm),
        prompt_name=prompt,
        use_images=use_images,
        redactor=redactor,
    )
    if candidate.parsed_ok:
        return candidate.packet, "llm", ""
    return None, "llm", candidate.rejection_reason


def generate_contract_invariants(
    fsm: AppFSM,
    raw_screens: dict[str, Any],
    app_prior: AppPrior | None = None,
    *,
    invariant_source: InvariantSource = "llm",
    llm: LlmClient | None = None,
    invariant_prompt: str = DEFAULT_INVARIANT_PROMPT,
    use_images: bool = False,
    redactor: PromptRedactor | None = None,
    redact_identifiers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Replace invariants from valid proposals; preserve them on generation failure."""
    if invariant_source == "llm" and llm is None:
        raise ValueError(f"invariant_source={invariant_source!r} requires an LLM client")

    evidence_items = build_all_invariant_evidence(fsm, raw_screens, app_prior)
    if redactor is None and invariant_source == "llm":
        from vigil.neuro.prompt_redaction import build_prompt_redactor

        redactor = build_prompt_redactor(
            fsm,
            evidence_items,
            extra_identifiers=redact_identifiers,
        )

    evaluator = DSLEvaluator()
    report: list[dict[str, Any]] = []

    for evidence in evidence_items:
        packet, origin, generation_error = _resolve_candidates(
            evidence,
            invariant_source,
            llm,
            invariant_prompt,
            use_images,
            redactor,
        )
        state = fsm.states[evidence.target_state_id]
        if packet is None:
            report.append(
                {
                    "state_id": state.state_id,
                    "state_name": evidence.target_state_name,
                    "source": origin,
                    "generation_status": "failed",
                    "generation_error": generation_error,
                    "observation_count": evidence.observation_count,
                    "invariants_admitted": [],
                    "invariants_attached": [invariant.expr for invariant in state.invariant_specs],
                    "rejected": [],
                }
            )
            continue
        admitted: list[StateInvariant] = []
        admitted_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        admitted_expressions: set[str] = set()

        for candidate in packet.candidates:
            result = admit_state_invariant_candidate(
                candidate,
                evidence,
                evaluator=evaluator,
            )
            row = {
                "kind": candidate.kind,
                "predicate": candidate.predicate.model_dump(mode="json"),
                "compiled_expr": result.compiled_expr,
                "reason": result.reason,
            }
            if result.invariant is None:
                rejected_rows.append(row)
                continue
            if result.compiled_expr in admitted_expressions:
                row["reason"] = "duplicate executable invariant"
                rejected_rows.append(row)
                continue
            admitted_expressions.add(result.compiled_expr)
            admitted.append(result.invariant)
            admitted_rows.append(row)

        state.invariant_specs = admitted
        report.append(
            {
                "state_id": evidence.target_state_id,
                "state_name": evidence.target_state_name,
                "source": origin,
                "generation_status": "succeeded",
                "generation_error": "",
                "observation_count": evidence.observation_count,
                "invariants_admitted": admitted_rows,
                "invariants_attached": [invariant.expr for invariant in admitted],
                "rejected": rejected_rows,
            }
        )

    return report


def write_invariant_generation_report(report: list[dict[str, Any]], path: Path) -> None:
    """Write a JSON report at the caller-selected path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

"""Contract-first guard generation pipeline (step 4).

Ties the guard stages together over an existing ``AppFSM``:

    build_all_guard_evidence
      -> group transitions by source_state_id
      -> per source state, bounded attempt loop:
           one per-transition LLM candidate
           -> admit_source_state_batch (per-contract admission + coherence)
      -> atomic per-state attachment on full success

The **neural generation unit is one transition**; the **symbolic admission unit is all
outgoing transitions of one source state**; the **attachment unit is one complete source-state
batch**. This does not mutate the FSM graph, state identity, transition evidence, or provenance.

Each attempt produces one LLM-typed :class:`GuardContract` per transition, up to
``guard_max_attempts`` whole-state attempts, with the prior attempt's admission diagnostics
injected into each retry prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vigil.models.fsm import canonical_transition_action
from vigil.neuro.guard_contract_llm import DEFAULT_GUARD_PROMPT, generate_llm_guard_candidate
from vigil.neuro.guard_evidence import build_all_guard_evidence
from vigil.neuro.guard_state_admission import (
    StateBatchDiagnostics,
    StateBatchOutcome,
    StateGuardCandidate,
    admit_source_state_batch,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vigil.core.llm_client import LlmClient
    from vigil.models.fsm import AppFSM, Transition
    from vigil.neuro.app_prior import AppPrior
    from vigil.neuro.guard_evidence import GuardEvidence
    from vigil.neuro.prompt_redaction import PromptRedactor

DEFAULT_GUARD_MAX_ATTEMPTS = 3


@dataclass
class GuardGenerationOutcome:
    """Explicit aggregate result of one guard-generation run.

    ``succeeded`` is ``True`` iff every source state's batch was admitted and atomically
    attached. ``report`` is the per-transition row list (sorted by ``transition_index``).
    """

    report: list[dict[str, Any]]
    succeeded: bool
    state_outcomes: list[StateBatchOutcome] = field(default_factory=list)


def _action_summary(action: dict[str, Any]) -> dict[str, Any]:
    visible_action = canonical_transition_action(action)
    return {
        "type": visible_action.get("type", ""),
        "target": visible_action.get("target", ""),
        "target_text": visible_action.get("target_text", ""),
        "target_resource_id": visible_action.get(
            "target_resource_id", visible_action.get("resource_id", "")
        ),
    }


def _generate_candidates(
    *,
    indices: list[int],
    evidence_items: list[GuardEvidence],
    llm: LlmClient,
    guard_prompt: str,
    guard_use_images: bool,
    redactor: PromptRedactor | None,
    retry_diagnostics: dict[str, Any] | None,
) -> list[StateGuardCandidate]:
    """Build one :class:`StateGuardCandidate` per outgoing transition (one call each)."""
    candidates: list[StateGuardCandidate] = []
    for index in indices:
        evidence = evidence_items[index]
        candidate = generate_llm_guard_candidate(
            evidence,
            llm,
            prompt_name=guard_prompt,
            use_images=guard_use_images,
            redactor=redactor,
            retry_diagnostics=retry_diagnostics,
        )
        candidates.append(
            StateGuardCandidate(
                transition_index=index,
                evidence=evidence,
                contract=candidate.contract,
                generation_failed=not candidate.parsed_ok,
                failure_reason=candidate.rejection_reason,
            )
        )
    return candidates


def _batch_reason(diagnostics: StateBatchDiagnostics) -> str:
    parts: list[str] = []
    if diagnostics.generation_failures:
        parts.append(f"{len(diagnostics.generation_failures)} generation failure(s)")
    if diagnostics.admission_failures:
        parts.append(f"{len(diagnostics.admission_failures)} admission failure(s)")
    if diagnostics.coherence_conflicts:
        names = ", ".join(conflict.slot_name for conflict in diagnostics.coherence_conflicts)
        parts.append(f"slot fingerprint conflict(s): {names}")
    return "; ".join(parts) if parts else "admitted"


def _report_rows(
    transitions: dict[int, Transition],
    indices: list[int],
    candidates: list[StateGuardCandidate],
    outcome: StateBatchOutcome,
) -> dict[int, dict[str, Any]]:
    """One admitted/rejected row per transition of the source state."""
    rows: dict[int, dict[str, Any]] = {}
    candidate_by_index = {candidate.transition_index: candidate for candidate in candidates}
    if outcome.admitted:
        for result in outcome.results:
            index = result.transition_index
            transition = transitions[index]
            rows[index] = {
                "transition_index": index,
                "source": transition.source,
                "target": transition.target,
                "action": _action_summary(transition.action),
                "guard_origin": "llm",
                "kind": result.contract.kind.value,
                "status": "admitted",
                "reason": result.result.reason,
                "rejected_predicates": list(result.result.rejected_predicates),
                "contract": result.contract.model_dump(mode="json"),
                "guard": result.guard,
            }
        return rows

    failures: dict[int, dict[str, Any]] = {}
    for failure in (
        *outcome.diagnostics.generation_failures,
        *outcome.diagnostics.admission_failures,
    ):
        failures[int(failure["transition_index"])] = dict(failure)
    batch_reason = _batch_reason(outcome.diagnostics)
    for index in indices:
        transition = transitions[index]
        candidate = candidate_by_index[index]
        failure = failures.get(index)
        reason = (
            str(failure["reason"])
            if failure is not None
            else f"withheld: source-state batch rejected ({batch_reason})"
        )
        rows[index] = {
            "transition_index": index,
            "source": transition.source,
            "target": transition.target,
            "action": _action_summary(transition.action),
            "guard_origin": "llm",
            "kind": candidate.contract.kind.value,
            "status": "rejected",
            "reason": reason,
            "rejected_predicates": (
                [str(item) for item in failure.get("rejected_predicates", [])]
                if failure is not None
                else []
            ),
            "contract": candidate.contract.model_dump(mode="json"),
            "guard": None,
        }
    return rows


def _generate_one_source_state(
    *,
    source_state_id: str,
    indices: list[int],
    fsm: AppFSM,
    evidence_items: list[GuardEvidence],
    llm: LlmClient,
    guard_prompt: str,
    guard_use_images: bool,
    redactor: PromptRedactor | None,
    guard_max_attempts: int,
) -> tuple[StateBatchOutcome, dict[int, dict[str, Any]]]:
    """Run the bounded attempt loop for one source state and attach on success."""
    transitions = {index: fsm.transitions[index] for index in indices}
    retry_diagnostics: dict[str, Any] | None = None

    for _attempt in range(guard_max_attempts):
        candidates = _generate_candidates(
            indices=indices,
            evidence_items=evidence_items,
            llm=llm,
            guard_prompt=guard_prompt,
            guard_use_images=guard_use_images,
            redactor=redactor,
            retry_diagnostics=retry_diagnostics,
        )
        outcome = admit_source_state_batch(candidates, source_state_id=source_state_id)
        if outcome.admitted:
            break
        retry_diagnostics = outcome.diagnostics.as_dict()

    if outcome.admitted:
        for result in outcome.results:
            transition = transitions[result.transition_index]
            transition.guard_contract = result.contract
            transition.guard = result.guard
    # Otherwise leave every transition untouched (prior contracts preserved).

    rows = _report_rows(transitions, indices, candidates, outcome)
    return outcome, rows


def generate_contract_guards(
    fsm: AppFSM,
    raw_screens: dict[str, Any],
    app_prior: AppPrior | None = None,
    *,
    llm: LlmClient | None = None,
    guard_prompt: str = DEFAULT_GUARD_PROMPT,
    guard_use_images: bool = False,
    redactor: PromptRedactor | None = None,
    redact_identifiers: list[str] | None = None,
    guard_max_attempts: int = DEFAULT_GUARD_MAX_ATTEMPTS,
) -> GuardGenerationOutcome:
    """Generate, admit, and atomically attach source-state guard batches across ``fsm``.

    Transitions are grouped by ``source_state_id``. Each source state runs a bounded attempt
    loop (up to ``guard_max_attempts``). Each attempt generates exactly one LLM candidate per
    outgoing transition, then :func:`admit_source_state_batch` runs per-contract admission plus
    the same-name slot-fingerprint coherence invariant. On full success the batch is attached
    atomically; otherwise nothing is attached (prior contracts are preserved) and diagnostics
    feed the next attempt's prompts. The FSM graph, state identity, transition evidence, and
    provenance are left unchanged.
    """
    if guard_max_attempts <= 0:
        raise ValueError(f"guard_max_attempts must be > 0, got {guard_max_attempts}")
    if llm is None:
        raise ValueError("generate_contract_guards requires an LLM client")

    evidence_items = build_all_guard_evidence(fsm, raw_screens, app_prior)
    if redactor is None:
        from vigil.neuro.prompt_redaction import build_prompt_redactor

        redactor = build_prompt_redactor(fsm, evidence_items, extra_identifiers=redact_identifiers)

    # Deterministic source-state grouping (first-seen order; index order within a group).
    groups: dict[str, list[int]] = {}
    for index, transition in enumerate(fsm.transitions):
        groups.setdefault(transition.source, []).append(index)

    rows_by_index: dict[int, dict[str, Any]] = {}
    state_outcomes: list[StateBatchOutcome] = []
    succeeded = True

    for source_state_id, indices in groups.items():
        outcome, rows = _generate_one_source_state(
            source_state_id=source_state_id,
            indices=indices,
            fsm=fsm,
            evidence_items=evidence_items,
            llm=llm,
            guard_prompt=guard_prompt,
            guard_use_images=guard_use_images,
            redactor=redactor,
            guard_max_attempts=guard_max_attempts,
        )
        rows_by_index.update(rows)
        state_outcomes.append(outcome)
        if not outcome.admitted:
            succeeded = False

    report = [rows_by_index[index] for index in sorted(rows_by_index)]
    return GuardGenerationOutcome(
        report=report,
        succeeded=succeeded,
        state_outcomes=state_outcomes,
    )


def write_guard_generation_report(report: list[dict[str, Any]], path: Path) -> None:
    """Write a guard-generation report as JSON at the caller-selected path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

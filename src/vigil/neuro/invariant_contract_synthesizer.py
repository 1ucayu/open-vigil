"""Minimal deterministic state-invariant candidate synthesis.

The deterministic source emits only stable title anchors established by repeated runtime
observations. Rich semantic status, selected state, and container-shape candidates belong
to the typed LLM path and still pass the same deterministic evidence-admission gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vigil.models.guard import PredicateSpec, ValueRef
from vigil.models.invariant_candidate import (
    InvariantCandidatePacket,
    StateInvariantCandidate,
)
from vigil.neuro.guard_registry import WidgetRole, elements_of

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vigil.neuro.guard_registry import WidgetRegistryEntry
    from vigil.neuro.invariant_evidence import InvariantEvidence


def _observed_text(resource_id: str, observations: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for observation in observations:
        match = next(
            (
                element
                for element in elements_of(observation)
                if str(element.get("resource_id") or "") == resource_id
            ),
            None,
        )
        if match is None:
            return []
        values.append(str(match.get("text") or "").strip())
    return values


def _stable_title_candidate(
    entry: WidgetRegistryEntry,
    observations: list[dict[str, Any]],
) -> StateInvariantCandidate | None:
    if entry.role is not WidgetRole.TITLE or not entry.resource_id:
        return None
    values = _observed_text(entry.resource_id, observations)
    if not values or not values[0] or len(set(values)) != 1:
        return None
    return StateInvariantCandidate(
        kind="stable_label",
        predicate=PredicateSpec(
            predicate_type="read",
            element=entry.alias,
            property="text",
            operator="==",
            expected=ValueRef(kind="literal", value=values[0]),
            source="deterministic",
        ),
        source="deterministic",
    )


def synthesize_invariant_candidates(evidence: InvariantEvidence) -> InvariantCandidatePacket:
    """Return repeated-observation stable title candidates for one state."""
    candidates = [
        candidate
        for entry in evidence.arrival_registry.entries.values()
        if (candidate := _stable_title_candidate(entry, evidence.observations)) is not None
    ]
    return InvariantCandidatePacket(candidates=candidates)

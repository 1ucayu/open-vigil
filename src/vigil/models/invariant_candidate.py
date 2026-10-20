"""Typed state-invariant candidate IR.

Invariant generation proposes one typed predicate at a time. Deterministic admission
resolves its source-state widget, validates the narrow invariant vocabulary, compiles it
once, and replays it over every runtime observation assigned to the state.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vigil.models.guard import PredicateSpec

InvariantKind = Literal[
    "stable_label",
    "container_shape",
    "form_status",
    "status",
    "semantic_role",
    "unknown",
]


class StateInvariantCandidate(BaseModel):
    """One typed, not-yet-admitted current-state predicate."""

    model_config = ConfigDict(extra="forbid")

    kind: InvariantKind
    predicate: PredicateSpec
    source: str = "generated"


class InvariantCandidatePacket(BaseModel):
    """Candidate list for one already-built abstract state."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[StateInvariantCandidate] = Field(default_factory=list)

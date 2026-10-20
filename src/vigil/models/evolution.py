"""Canonical self-evolution taxonomy for the Vigil EFSM.

Coverage expansion, abstraction refinement, and contract enrichment update the
application model. Evidence enrichment records corroborating observations without
changing its supported behavior. Transport channels and attempt outcomes remain
separate from these operations.
"""

from __future__ import annotations

from enum import StrEnum


class EvolutionOperation(StrEnum):
    """Model changes and evidence bookkeeping for the per-app EFSM ``M_A``."""

    COVERAGE_EXPANSION = "coverage_expansion"
    ABSTRACTION_REFINEMENT = "abstraction_refinement"
    CONTRACT_ENRICHMENT = "contract_enrichment"
    EVIDENCE_ENRICHMENT = "evidence_enrichment"


class EvolutionOutcome(StrEnum):
    """Factual result of attempting an evolution operation."""

    APPLIED = "applied"
    CORROBORATED = "corroborated"
    NOOP = "noop"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"

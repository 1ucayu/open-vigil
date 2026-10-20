"""Screen-to-FSM state mapping via accessibility tree fingerprinting.

Maps the current device screen to a known FSM state using structural fingerprinting.
Falls back to a non-authoritative functional-hash scan for evidence-guided evolution when the
deterministic lookups fail.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from vigil.models.fsm import AppFSM
from vigil.models.state import RawScreen
from vigil.neuro.behavioral_signature import (
    compute_quotient_label,
    refinement_identity_hash,
    signature_hash,
)
from vigil.neuro.state_split import SECONDARY_MARKER as _REFINED_SECONDARY_MARKER
from vigil.neuro.state_split import secondary_feature_screen_hash


class LocateResult(StrEnum):
    """Outcome of a state localization attempt."""

    EXACT = "exact"
    CANONICAL_EQUIVALENT = "canonical_equivalent"
    SIMILAR = "similar"
    UNKNOWN = "unknown"


def is_known_location(result: LocateResult) -> bool:
    """True for deterministic localization results usable by the verifier."""
    return result in {LocateResult.EXACT, LocateResult.CANONICAL_EQUIVALENT}


class StateLocation(BaseModel):
    """Result of mapping a screen to an FSM state.

    Attributes:
        result: Whether the match was exact, canonical-equivalent, similar, or
            unknown. This is the only trust signal: the verifier keys off
            ``is_known_location(result)``.
        state_id: The matched FSM state ID (None if unknown).
        matched_fingerprint: The fingerprint that was matched against.
    """

    result: LocateResult
    state_id: str | None = None
    matched_fingerprint: str | None = None


# A live RawScreen truncates its structural/functional hashes to 12 hex chars,
# while the offline builder stores 16-char hashes with the same prefix.
_LIVE_HASH_LENGTH = 12


class StateLocator:
    """Maps a runtime screen to a known FSM state via fingerprint matching.

    Pre-builds a fingerprint index for O(1) exact lookup. Falls back to
    AppFSM.find_state_by_fingerprint() for a functional-hash scan.

    Args:
        fsm: The app's FSM to localize against.
    """

    def __init__(self, fsm: AppFSM) -> None:
        self._fsm = fsm
        self._fp_index: dict[str, list[str]] = {}
        self._canonical_index: dict[str, list[str]] = {}
        # Refined-sibling index keyed by (base_fingerprint, secondary_hash) so
        # a live screen with the same secondary signature but a DIFFERENT base
        # structural fingerprint cannot falsely localize to a refined sibling.
        # The builder guarantees every refined fingerprint encodes a single
        # concrete secondary hash (see _refine_conflicting_successors policy).
        self._refined_fp_index: dict[tuple[str, str], str] = {}
        # Runtime Abstraction-refinement identity, resolved atomically and scoped per
        # split-source family: (family, refinement_signature) -> state. Each refined child's
        # functional/structural/canonical hashes map to its family; a live screen matching a
        # family localizes only by a complete refinement-identity match (never by a component
        # hash alone), and a near-miss or ambiguity returns UNKNOWN.
        self._refinement_index: dict[tuple[str, str], str] = {}
        self._family_lookup: dict[str, set[str]] = {}
        for state in fsm.states.values():
            identity = state.identity
            if identity.refinement_signature and identity.refinement_family:
                key = (identity.refinement_family, identity.refinement_signature)
                existing = self._refinement_index.get(key)
                # Poison a colliding (family, signature) pair so it localizes UNKNOWN.
                if existing not in (None, state.state_id):
                    self._refinement_index[key] = ""
                else:
                    self._refinement_index[key] = state.state_id
                for value in (
                    identity.functional_hash,
                    identity.structural_hash,
                    identity.canonical_equivalence_hash,
                    identity.refinement_signature,
                ):
                    if value:
                        self._family_lookup.setdefault(value, set()).add(identity.refinement_family)
                continue
            self._add_fingerprint_match(identity.functional_hash, state.state_id)
            self._index_refined_sibling(identity.functional_hash, state.state_id)
            # Schema-v6: a quotiented state covers many observed live-screen identities
            # but stores one representative in ``functional_hash``. Index the rest so a
            # member screen EXACT-matches its own state instead of falling through to a
            # canonical bucket that several states may share. Empty on v5 payloads, so
            # this is a no-op for un-migrated FSMs; a fingerprint claimed by two states
            # poisons its slot to UNKNOWN through the existing multi-hit rule.
            for member_fingerprint in identity.member_fingerprints:
                if member_fingerprint and member_fingerprint != identity.functional_hash:
                    self._add_fingerprint_match(member_fingerprint, state.state_id)
            if identity.structural_hash:
                self._add_fingerprint_match(identity.structural_hash, state.state_id)
                self._index_refined_sibling(identity.structural_hash, state.state_id)
                # The builder stores 16-char structural hashes; a live RawScreen produces
                # the 12-char prefix. Index the prefix so a live screen EXACT-matches a
                # builder-produced state (ambiguous prefixes collapse to UNKNOWN safely).
                if (
                    _REFINED_SECONDARY_MARKER not in identity.structural_hash
                    and len(identity.structural_hash) > _LIVE_HASH_LENGTH
                ):
                    self._add_fingerprint_match(
                        identity.structural_hash[:_LIVE_HASH_LENGTH], state.state_id
                    )
            if identity.canonical_equivalence_hash:
                self._add_canonical_match(
                    identity.canonical_equivalence_hash,
                    state.state_id,
                )

    def _add_fingerprint_match(self, fingerprint: str, state_id: str) -> None:
        matches = self._fp_index.setdefault(fingerprint, [])
        if state_id not in matches:
            matches.append(state_id)

    def _add_canonical_match(self, canonical_hash: str, state_id: str) -> None:
        matches = self._canonical_index.setdefault(canonical_hash, [])
        if state_id not in matches:
            matches.append(state_id)

    def _index_refined_sibling(self, fingerprint: str, state_id: str) -> None:
        if _REFINED_SECONDARY_MARKER not in fingerprint:
            return
        base_fp, _, secondary_hash = fingerprint.rpartition(_REFINED_SECONDARY_MARKER)
        if not base_fp or not secondary_hash:
            return
        # Builder stores 16-char hashes; RawScreen.get_structural_fingerprint
        # returns 12-char hashes (same prefix, narrower truncation). Normalize
        # the index key to the live-screen length so locate() can match.
        live_base_fp = base_fp[:_LIVE_HASH_LENGTH]
        key = (live_base_fp, secondary_hash)
        existing = self._refined_fp_index.get(key)
        if existing is not None and existing != state_id:
            # Two refined siblings with the same (base_fp, secondary_hash) would
            # be indistinguishable; mark the slot poisoned by storing an empty
            # string and let lookups fall through to UNKNOWN/SIMILAR.
            self._refined_fp_index[key] = ""
            return
        self._refined_fp_index[key] = state_id

    def locate(self, screen: RawScreen) -> StateLocation:
        """Locate a live screen in the FSM.

        Args:
            screen: The current screen with parsed UI elements.

        Returns:
            StateLocation with the match result and state_id.
        """
        fp = screen.get_structural_fingerprint()

        # Compute both the ordinary deterministic result and the family-scoped refinement
        # result, then reconcile: a weak shared component hash must not let a refined family
        # override a stronger unique canonical match, and a refined child's own screen must
        # still resolve to that child.
        refined = self._locate_refined(screen, fp)
        ordinary = self._locate_ordinary(screen, fp)

        refined_id = (
            refined.state_id
            if (refined is not None and refined.result is LocateResult.EXACT)
            else None
        )
        ordinary_definite = ordinary if is_known_location(ordinary.result) else None
        ordinary_id = ordinary_definite.state_id if ordinary_definite is not None else None

        if refined_id is not None and ordinary_id is not None and refined_id != ordinary_id:
            # Disagreement: the state whose stored canonical hash equals the live canonical
            # hash wins; if neither or both do, refuse to guess.
            live_canonical = raw_screen_canonical_equivalence_hash(screen)
            refined_canon = (
                self._fsm.states[refined_id].identity.canonical_equivalence_hash == live_canonical
            )
            ordinary_canon = (
                self._fsm.states[ordinary_id].identity.canonical_equivalence_hash == live_canonical
            )
            if refined_canon and not ordinary_canon:
                return refined
            if ordinary_canon and not refined_canon:
                return ordinary_definite
            return StateLocation(result=LocateResult.UNKNOWN, matched_fingerprint=fp)

        if refined_id is not None:
            return refined
        if ordinary_id is not None:
            return ordinary_definite
        # A refined family near-miss (family matched, no atomic signature) with no definite
        # ordinary match is genuinely unknown — never fall through to weak similarity.
        if refined is not None:
            return StateLocation(result=LocateResult.UNKNOWN, matched_fingerprint=fp)
        return ordinary

    def _locate_ordinary(self, screen: RawScreen, fp: str) -> StateLocation:
        """Ordinary deterministic localization over the non-refined indexes: exact fingerprint
        → candidate fingerprints → offline refined-sibling → canonical → non-authoritative
        functional-hash fallback."""
        location = self._locate_exact_by_fingerprint(fp)
        if location.result is not LocateResult.UNKNOWN:
            return location

        for candidate_fp in self._candidate_fingerprints(screen, fp)[1:]:
            location = self._locate_exact_by_fingerprint(candidate_fp)
            if location.result is not LocateResult.UNKNOWN:
                return location

        # Offline refined-sibling fallback: require the live screen's structural fingerprint
        # to match the refined state's *base* fingerprint AND its concrete secondary hash.
        secondary_hash = self._secondary_feature_signature_hash(screen)
        refined_state_id = self._refined_fp_index.get((fp, secondary_hash))
        if refined_state_id:
            refined = self._fsm.states.get(refined_state_id)
            matched = refined.identity.structural_hash if refined else None
            return StateLocation(
                result=LocateResult.EXACT,
                state_id=refined_state_id,
                matched_fingerprint=matched
                or (refined.identity.functional_hash if refined else fp),
            )

        canonical_location = self._locate_by_canonical_hash(screen)
        if canonical_location.result is not LocateResult.UNKNOWN:
            return canonical_location
        # An ambiguous canonical bucket (the live label matched, but several states share
        # it) used to return UNKNOWN here, which also swallowed the SIMILAR scan below.
        # Fall through instead. SIMILAR stays non-authoritative -- the verifier still
        # answers UNCERTAIN and never ALLOW -- so nothing becomes less safe, but evolution
        # and diagnostics see the candidate state instead of a bare UNKNOWN.

        for candidate_fp in self._candidate_fingerprints(screen, fp):
            location = self._locate_fallback_by_functional_hash(candidate_fp)
            if location.result is LocateResult.SIMILAR:
                return location

        return StateLocation(
            result=LocateResult.UNKNOWN,
            matched_fingerprint=fp,
        )

    def _locate_refined(self, screen: RawScreen, structural_fp: str) -> StateLocation | None:
        """Atomic, family-scoped localization for runtime Abstraction-refinement states.

        Returns ``None`` when the screen belongs to no refined family (use the normal
        paths). Otherwise it requires a complete refinement-identity match within a family:
        exactly one matched state → EXACT; a near-miss (family matched but no signature
        match), a poisoned slot, or an ambiguous family/state → UNKNOWN.
        """
        if not self._family_lookup:
            return None
        candidate_hashes = set(self._candidate_fingerprints(screen, structural_fp))
        candidate_hashes.add(raw_screen_canonical_equivalence_hash(screen))
        families: set[str] = set()
        for value in candidate_hashes:
            families |= self._family_lookup.get(value, set())
        if not families:
            return None
        refinement_id = refinement_identity_hash(raw_screen_to_quotient_input(screen))
        matched = {self._refinement_index.get((family, refinement_id)) for family in families}
        matched.discard(None)
        matched.discard("")
        if len(matched) == 1:
            return StateLocation(
                result=LocateResult.EXACT,
                state_id=next(iter(matched)),
                matched_fingerprint=structural_fp,
            )
        return StateLocation(
            result=LocateResult.UNKNOWN,
            matched_fingerprint=structural_fp,
        )

    def _candidate_fingerprints(self, screen: RawScreen, structural_fp: str) -> list[str]:
        """Return live-screen identity candidates from strongest local evidence.

        FSMs may store either the raw structural fingerprint or the hybrid
        functional state id as ``identity.functional_hash``. Try both without
        weakening the lookup: every candidate still has to hit the FSM's exact
        fingerprint index.
        """
        candidates = [structural_fp]
        app_package = self._fsm.app_package
        if app_package:
            hybrid = screen.get_hybrid_state_id(app_package)
            if hybrid and hybrid != structural_fp:
                candidates.append(hybrid)
        return candidates

    def _locate_exact_by_fingerprint(self, fingerprint: str) -> StateLocation:
        """Exact lookup only; does not route to the functional-hash fallback."""
        state_ids = self._fp_index.get(fingerprint)
        if state_ids is not None and len(state_ids) == 1:
            return StateLocation(
                result=LocateResult.EXACT,
                state_id=state_ids[0],
                matched_fingerprint=fingerprint,
            )
        if state_ids is not None:
            return StateLocation(
                result=LocateResult.UNKNOWN,
                matched_fingerprint=fingerprint,
            )
        return StateLocation(
            result=LocateResult.UNKNOWN,
            matched_fingerprint=None,
        )

    def _locate_fallback_by_functional_hash(self, fingerprint: str) -> StateLocation:
        """Non-authoritative functional-hash scan for evolution only.

        This is an exact scan over ``identity.functional_hash`` (first match wins),
        not fuzzy matching; the verifier treats its result as UNCERTAIN.
        """
        fallback_id = self._fsm.find_state_by_fingerprint(fingerprint)
        if fallback_id is not None:
            return StateLocation(
                result=LocateResult.SIMILAR,
                state_id=fallback_id,
                matched_fingerprint=fingerprint,
            )

        # No match
        return StateLocation(
            result=LocateResult.UNKNOWN,
            matched_fingerprint=fingerprint,
        )

    def _locate_by_canonical_hash(self, screen: RawScreen) -> StateLocation:
        canonical_hash = raw_screen_canonical_equivalence_hash(screen)
        state_ids = self._canonical_index.get(canonical_hash)
        if state_ids is not None and len(state_ids) == 1:
            return StateLocation(
                result=LocateResult.CANONICAL_EQUIVALENT,
                state_id=state_ids[0],
                matched_fingerprint=canonical_hash,
            )
        if state_ids is not None:
            return StateLocation(
                result=LocateResult.UNKNOWN,
                matched_fingerprint=canonical_hash,
            )
        return StateLocation(
            result=LocateResult.UNKNOWN,
            matched_fingerprint=None,
        )

    @staticmethod
    def _secondary_feature_signature_hash(screen: RawScreen) -> str:
        return secondary_feature_screen_hash(screen)


def raw_screen_to_quotient_input(screen: RawScreen) -> dict[str, object]:
    """Adapt a live RawScreen to the trace-screen dict shape used by the quotient."""
    return {
        "activity_name": screen.activity_name or "",
        "package_name": screen.package_name or "",
        "metadata": dict(screen.metadata),
        "elements": [element.model_dump(mode="json") for element in screen.elements],
    }


def raw_screen_canonical_equivalence_hash(screen: RawScreen) -> str:
    """Deterministic runtime canonical-equivalence hash for a live screen."""
    return signature_hash(compute_quotient_label(raw_screen_to_quotient_input(screen)))

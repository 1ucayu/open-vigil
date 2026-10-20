"""Shared state-split identity helpers (Abstraction refinement).

The *secondary-feature signature* distinguishes raw screens that share a primary
fingerprint but observably behave differently. Offline FSM construction
(``fsm_builder``) mints ``::secondary:<hash>`` split fingerprints from it during a
verifier-preserving state split, and the runtime :class:`~vigil.symbolic.state_locator.StateLocator`
recomputes the same hash to localize split states. Both call the functions here so
the split identity computed offline and the identity recomputed at runtime always
agree.

Only app-neutral features are used: activity, modal boundary, sorted non-editable
text anchors, and sorted content-description anchors. Editable-field text is
excluded because user input is volatile. Literal list/row values that merely happen
to be non-editable are not treated as inherently stable; the callers additionally
require the signature to be consistent across every observation assigned to a child
before they act on it.
"""

from __future__ import annotations

import hashlib
from typing import Any

# Marker separating a base fingerprint from its split-specific secondary hash.
SECONDARY_MARKER = "::secondary:"


def _screen_fields(screen: Any) -> tuple[str, bool, list[dict[str, Any]]]:
    """Normalize a trace-screen dict or a ``RawScreen`` into ``(activity, has_modal, elements)``.

    Trace dicts prefer ``interactable_elements`` then ``elements`` (offline shape).
    A ``RawScreen`` is duck-typed via its ``activity_name``/``metadata``/``elements``
    attributes and its elements are dumped to the same dict shape the offline path
    reads, so both inputs produce an identical signature.
    """
    if isinstance(screen, dict):
        metadata = screen.get("metadata", {})
        elements = screen.get("interactable_elements", screen.get("elements", []))
        has_modal = bool(metadata.get("has_modal")) if isinstance(metadata, dict) else False
        return (screen.get("activity_name") or "", has_modal, list(elements))
    elements = [element.model_dump(mode="json") for element in screen.elements]
    return (screen.activity_name or "", bool(screen.metadata.get("has_modal")), elements)


def secondary_feature_signature(screen: Any) -> tuple[Any, ...]:
    """Stable secondary features distinguishing screens with the same primary fingerprint.

    Combines activity, modal flag, sorted non-editable text anchors, and sorted
    content-description anchors. Editable-field text is excluded (user input is
    volatile).
    """
    activity, has_modal, elements = _screen_fields(screen)
    text_anchors = sorted(
        {
            (el.get("text") or "").strip()
            for el in elements
            if (el.get("text") or "").strip() and not el.get("is_editable")
        }
    )
    desc_anchors = sorted(
        {
            (el.get("content_description") or "").strip()
            for el in elements
            if (el.get("content_description") or "").strip()
        }
    )
    return (activity, has_modal, tuple(text_anchors), tuple(desc_anchors))


def secondary_feature_signature_hash(signature: tuple[Any, ...]) -> str:
    """12-hex stable hash of a secondary-feature signature tuple."""
    return hashlib.sha256(repr(signature).encode()).hexdigest()[:12]


def secondary_feature_screen_hash(screen: Any) -> str:
    """Signature hash of a single screen (trace dict or ``RawScreen``)."""
    return secondary_feature_signature_hash(secondary_feature_signature(screen))


def secondary_feature_group_hash(screen_ids: set[str], raw_screens: dict[str, Any]) -> str:
    """Group hash over the secondary signatures of a set of screens.

    A single distinct signature returns its own hash; multiple distinct signatures
    fold deterministically into one group hash.
    """
    signature_hashes = sorted(
        secondary_feature_screen_hash(raw_screens.get(sid, {})) for sid in screen_ids
    )
    if len(signature_hashes) == 1:
        return signature_hashes[0]
    return hashlib.sha256(repr(tuple(signature_hashes)).encode()).hexdigest()[:12]


def with_secondary_feature_hash(fingerprint: str | None, secondary_hash: str) -> str | None:
    """Attach a split-specific secondary hash to a state fingerprint."""
    if not fingerprint:
        return fingerprint
    base = fingerprint.split(SECONDARY_MARKER, 1)[0]
    return f"{base}{SECONDARY_MARKER}{secondary_hash}"

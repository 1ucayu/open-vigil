"""Stable component selectors for UI elements.

A *selector* is a JSON-serializable dict of stable, capture-independent
properties that re-identifies an element across captures even when bounds,
element_id, or scroll offset change. Used by ``Action.target_selector`` to
make actions persistent across replay sessions.

Selector fields:

    {
        "resource_id": "...",
        "text": "...",
        "content_description": "...",
        "class_name": "...",
        "nearby_text": "...",            # descendant android:id/title text
        "row_discriminator": "...",      # descendant text unique among sibling rows
        "ancestor_chain": ["RecyclerView#recycler_view", ...],
    }

Identity priority (strongest first): resource_id > text > content_description
> nearby_text > row_discriminator > class_name + ancestor_chain. An element
with none of the first five has *no stable identity*
(``selector_has_stable_identity`` → False); class+ancestor is a weak last
resort used only during resolution.

``row_discriminator`` is RESOLUTION-ONLY. It is deliberately absent from
``fsm._SELECTOR_IDENTITY_FIELDS``, the closed allowlist that
``_selector_signature`` and ``canonical_transition_action`` iterate, so it can
never reach ``canonical_action_key``. A list row that carries no identity of
its own therefore still keys to one row-agnostic edge — the discriminator only
says *which* row a given proposal meant, exactly as capture-local bounds and
depth are already excluded from identity.
"""

from __future__ import annotations

from typing import Any

from vigil.core.platform_priors import get_repeated_row_containers
from vigil.models.state import UIElement, _normalize_dynamic

_MAX_ANCESTOR_DEPTH = 5


def _short_rid_tail(rid: str | None) -> str:
    """Return the segment after ``:id/`` if present, else the raw string."""
    if not rid:
        return ""
    if ":id/" in rid:
        return rid.split(":id/", 1)[1]
    return rid


def _class_leaf(class_name: str | None) -> str:
    """Return the final dotted segment of a class name (``android.widget.Button`` → ``Button``)."""
    if not class_name:
        return ""
    return class_name.rsplit(".", 1)[-1]


def _descendant_title_text(element: UIElement, by_id: dict[str, UIElement]) -> str:
    """Return the normalized text of the first ``android:id/title`` descendant.

    Mirrors ``vigil.neuro.explorer._descendant_title_text`` so selector
    building and explorer enumeration agree on the Preference-row label.
    Returns ``""`` if no such descendant exists.
    """
    stack = list(element.children)
    seen: set[str] = set()
    while stack:
        cid = stack.pop()
        if cid in seen:
            continue
        seen.add(cid)
        child = by_id.get(cid)
        if child is None:
            continue
        if child.resource_id == "android:id/title" and child.text:
            return _normalize_dynamic(child.text)
        stack.extend(child.children)
    return ""


def is_anonymous_clickable(element: UIElement) -> bool:
    """True when an actionable element carries no identity of its own.

    Such an element cannot be named by any selector field, so it is exactly the
    case that needs a descendant-derived discriminator.
    """
    if not (element.is_clickable or element.is_long_clickable):
        return False
    return not (element.resource_id or element.text or element.content_description)


def _repeated_row_siblings(
    element: UIElement, by_id: dict[str, UIElement]
) -> list[UIElement] | None:
    """Return the sibling rows ``element`` competes with, or ``None`` if it is not a row.

    A row is an anonymous clickable whose nearest ancestor collection is one of
    the platform's repeated-row containers (``platform_priors``). Its siblings
    are the other anonymous clickables under that same container -- the set the
    discriminator has to tell it apart from.
    """
    containers = get_repeated_row_containers()
    cur_id = element.parent_id
    depth = 0
    while cur_id and depth < _MAX_ANCESTOR_DEPTH:
        depth += 1
        parent = by_id.get(cur_id)
        if parent is None:
            return None
        if _class_leaf(parent.class_name) in containers:
            siblings = [
                other
                for other in by_id.values()
                if other.element_id != element.element_id
                and other.class_name == element.class_name
                and is_anonymous_clickable(other)
                and _has_ancestor(other, parent.element_id, by_id)
            ]
            return siblings
        cur_id = parent.parent_id
    return None


def _has_ancestor(element: UIElement, ancestor_id: str, by_id: dict[str, UIElement]) -> bool:
    cur_id = element.parent_id
    depth = 0
    while cur_id and depth < _MAX_ANCESTOR_DEPTH:
        depth += 1
        if cur_id == ancestor_id:
            return True
        parent = by_id.get(cur_id)
        if parent is None:
            return False
        cur_id = parent.parent_id
    return False


def _subtree_texts(element: UIElement, by_id: dict[str, UIElement]) -> list[str]:
    """Normalized descendant texts/content-descriptions, in document order.

    Document order matters: it makes the chosen discriminator deterministic
    across captures of the same screen.
    """
    texts: list[str] = []
    seen: set[str] = set()

    stack = list(reversed(element.children))
    while stack:
        eid = stack.pop()
        if eid in seen:
            continue
        seen.add(eid)
        child = by_id.get(eid)
        if child is None:
            continue
        for raw in (child.text, child.content_description):
            if raw and raw.strip():
                normalized = _normalize_dynamic(raw.strip())
                if normalized not in texts:
                    texts.append(normalized)
        stack.extend(reversed(child.children))
    return texts


def discriminative_descendant_text(element: UIElement, elements: list[UIElement]) -> str | None:
    """Return the descendant text that tells ``element`` apart from its sibling rows.

    Resolution-only. This never enters ``canonical_action_key``: it is written to
    a selector field outside ``_SELECTOR_IDENTITY_FIELDS``, so a stored edge stays
    row-agnostic and one edge keeps covering every row. Its sole job is to say
    *which* of N identical rows a proposal meant.

    The naive "first descendant text" is wrong: a decorative label repeated on
    every row (an image's content-description, a shared caption) identifies
    nothing. A candidate is therefore accepted only when it is unique among the
    sibling rows' descendant texts.

    Returns ``None`` -- deliberately leaving the action unbindable -- when the
    element is not a repeated row, has no descendant text, or has none that
    discriminates. Guessing here would bind the agent's action to the wrong row,
    which is precisely the error the shield exists to prevent.
    """
    if not is_anonymous_clickable(element):
        return None
    by_id = {e.element_id: e for e in elements}
    siblings = _repeated_row_siblings(element, by_id)
    if siblings is None:
        return None

    own_texts = _subtree_texts(element, by_id)
    if not own_texts:
        return None
    sibling_texts = {text for sib in siblings for text in _subtree_texts(sib, by_id)}
    for candidate in own_texts:
        if candidate not in sibling_texts:
            return candidate
    return None


def _build_ancestor_chain(element: UIElement, by_id: dict[str, UIElement]) -> list[str]:
    """Walk parents up to ``_MAX_ANCESTOR_DEPTH`` and emit ``Class#rid_tail`` tokens.

    Tokens are ordered top-down (outermost ancestor first). An ancestor with
    no resource-id contributes just the class leaf (``Class#``). Useful as a
    weak last-resort discriminator when class alone is too generic.
    """
    chain: list[str] = []
    cur_id = element.parent_id
    while cur_id and len(chain) < _MAX_ANCESTOR_DEPTH:
        parent = by_id.get(cur_id)
        if parent is None:
            break
        token = f"{_class_leaf(parent.class_name)}#{_short_rid_tail(parent.resource_id)}"
        chain.append(token)
        cur_id = parent.parent_id
    chain.reverse()
    return chain


def build_component_selector(element: UIElement, elements: list[UIElement]) -> dict[str, Any]:
    """Build a stable selector dict for ``element`` using ``elements`` as context.

    Dynamic text is normalized (``_normalize_dynamic``) so numeric values,
    sizes, percentages, and timestamps in labels do not destabilize the
    selector. ``nearby_text`` is borrowed from the first ``android:id/title``
    descendant when the element itself has no usable text/content-desc —
    matches the Preference-row pattern where the label lives inside a
    clickable container.

    Args:
        element: The UI element to describe.
        elements: All sibling/ancestor elements on the same screen (used to
            resolve parents and descendants).

    Returns:
        A dict with the fields documented in the module docstring. Empty
        values are returned as empty strings or empty lists so the shape is
        stable for JSON serialization and downstream consumers.
    """
    by_id = {e.element_id: e for e in elements}

    own_text = _normalize_dynamic(element.text) if element.text else ""
    cd = element.content_description or ""
    rid = element.resource_id or ""
    nearby = ""
    if not own_text and not cd:
        nearby = _descendant_title_text(element, by_id)

    # Resolution-only. ``row_discriminator`` is deliberately absent from
    # ``fsm._SELECTOR_IDENTITY_FIELDS``, the closed allowlist that both
    # ``_selector_signature`` and ``canonical_transition_action`` iterate, so it
    # can never reach ``canonical_action_key``. A stored row edge therefore stays
    # row-agnostic and one edge keeps covering every row; this field only says
    # which row a given proposal meant.
    row_discriminator = ""
    if not own_text and not cd and not rid:
        row_discriminator = discriminative_descendant_text(element, elements) or ""

    return {
        "resource_id": rid,
        "text": own_text,
        "content_description": cd,
        "class_name": element.class_name or "",
        "nearby_text": nearby,
        "row_discriminator": row_discriminator,
        "ancestor_chain": _build_ancestor_chain(element, by_id),
    }


def selector_identity(selector: dict[str, Any]) -> str:
    """Stable identity string for a selector.

    Concatenates priority-ordered fields with ``|`` separators. Bounds and
    depth are excluded — they are capture-volatile. Empty fields contribute
    empty segments, so two selectors that differ only in bounds yield the
    same identity.
    """
    if not selector:
        return ""
    rid = selector.get("resource_id") or ""
    text = selector.get("text") or ""
    cd = selector.get("content_description") or ""
    nearby = selector.get("nearby_text") or ""
    cls = selector.get("class_name") or ""
    chain = selector.get("ancestor_chain") or []
    chain_str = ">".join(chain) if isinstance(chain, list) else str(chain)
    return "|".join([rid, text, cd, nearby, cls, chain_str])


def selector_has_stable_identity(selector: dict[str, Any]) -> bool:
    """True iff the selector carries at least one strong identity field.

    Strong fields: ``resource_id``, ``text``, ``content_description``,
    ``nearby_text``, ``row_discriminator``. ``class_name`` and
    ``ancestor_chain`` alone are too generic — e.g. dozens of
    ``android.widget.LinearLayout`` exist on most screens — so they don't count
    as stable identity even though they help during resolution.

    ``row_discriminator`` counts because it is, by construction, unique among
    the sibling rows it was derived from: it names exactly one row on the screen
    it came from. It is still resolution-only and never enters action identity.
    """
    if not selector:
        return False
    for key in (
        "resource_id",
        "text",
        "content_description",
        "nearby_text",
        "row_discriminator",
    ):
        if selector.get(key):
            return True
    return False


def find_element_by_selector(
    selector: dict[str, Any], elements: list[UIElement]
) -> UIElement | None:
    """Locate the element matching ``selector`` on ``elements``.

    Matching order:
      1. If ``resource_id`` is set, return the first element whose rid matches.
      2. Else if ``text`` is set, match (a) elements whose own normalized text
         equals it or (b) clickable containers with an ``android:id/title``
         descendant whose text matches (Preference-row fallback).
      3. Else if ``content_description`` is set, match by content-desc.
      4. Else if ``nearby_text`` is set, match a container whose
         ``android:id/title`` descendant text matches.
      5. Else if ``class_name`` and ``ancestor_chain`` are set, weak match
         by class leaf + ancestor token sequence.

    Returns ``None`` when nothing matches. The caller is expected to fall
    back to legacy descriptor matching.
    """
    if not selector or not elements:
        return None
    by_id = {e.element_id: e for e in elements}

    rid = selector.get("resource_id") or ""
    text = selector.get("text") or ""
    cd = selector.get("content_description") or ""
    nearby = selector.get("nearby_text") or ""
    cls = selector.get("class_name") or ""
    chain = selector.get("ancestor_chain") or []

    def class_matches(e: UIElement) -> bool:
        if not cls:
            return True
        if e.class_name == cls:
            return True
        return _class_leaf(e.class_name) == _class_leaf(cls)

    def chain_matches(e: UIElement) -> bool:
        if not chain:
            return True
        return _build_ancestor_chain(e, by_id) == list(chain)

    def has_interaction(e: UIElement) -> bool:
        return e.is_clickable or e.is_long_clickable or e.is_editable or e.is_checkable

    def own_text_matches(e: UIElement, value: str) -> bool:
        return (_normalize_dynamic(e.text or "") if e.text else "") == value

    def direct_or_sibling_title_matches(e: UIElement, value: str) -> bool:
        for cid in e.children:
            child = by_id.get(cid)
            if child is None:
                continue
            if child.resource_id == "android:id/title" and own_text_matches(child, value):
                return True

        if e.parent_id:
            parent = by_id.get(e.parent_id)
            if parent is not None:
                for sid in parent.children:
                    if sid == e.element_id:
                        continue
                    sibling = by_id.get(sid)
                    if sibling is None:
                        continue
                    if sibling.resource_id == "android:id/title" and own_text_matches(
                        sibling, value
                    ):
                        return True
        return False

    def descendant_has_better_nearby_match(e: UIElement, value: str) -> bool:
        stack = list(e.children)
        seen: set[str] = set()
        while stack:
            cid = stack.pop()
            if cid in seen:
                continue
            seen.add(cid)
            child = by_id.get(cid)
            if child is None:
                continue
            if has_interaction(child) and candidate_nearby_matches(child, value):
                return True
            stack.extend(child.children)
        return False

    def candidate_nearby_matches(e: UIElement, value: str) -> bool:
        if own_text_matches(e, value):
            return has_interaction(e)
        if not has_interaction(e):
            return False
        if descendant_has_better_nearby_match(e, value):
            return False
        return direct_or_sibling_title_matches(e, value) or (
            _descendant_title_text(e, by_id) == value
        )

    def candidate_matches_all_fields(e: UIElement) -> bool:
        if rid and e.resource_id != rid:
            return False
        if text and not (own_text_matches(e, text) or candidate_nearby_matches(e, text)):
            return False
        if cd and (e.content_description or "") != cd:
            return False
        if nearby and not candidate_nearby_matches(e, nearby):
            return False
        return class_matches(e) and chain_matches(e)

    def unique_or_none(candidates: list[UIElement]) -> UIElement | None:
        if len(candidates) == 1:
            return candidates[0]
        return None

    if rid:
        return unique_or_none(
            [e for e in elements if e.resource_id == rid and candidate_matches_all_fields(e)]
        )

    if text:
        own_matches = [
            e for e in elements if own_text_matches(e, text) and candidate_matches_all_fields(e)
        ]
        if own_matches:
            return unique_or_none(own_matches)
        # Preference-row fallback: a clickable container whose title
        # descendant matches.
        return unique_or_none(
            [
                e
                for e in elements
                if candidate_nearby_matches(e, text) and candidate_matches_all_fields(e)
            ]
        )

    if cd:
        return unique_or_none(
            [
                e
                for e in elements
                if (e.content_description or "") == cd and candidate_matches_all_fields(e)
            ]
        )

    if nearby:
        return unique_or_none(
            [
                e
                for e in elements
                if candidate_nearby_matches(e, nearby) and candidate_matches_all_fields(e)
            ]
        )

    if cls and chain:
        cls_leaf = _class_leaf(cls)
        return unique_or_none(
            [
                e
                for e in elements
                if _class_leaf(e.class_name) == cls_leaf
                and _build_ancestor_chain(e, by_id) == list(chain)
            ]
        )

    return None

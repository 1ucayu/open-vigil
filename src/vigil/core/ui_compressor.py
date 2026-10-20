"""Plain-text UI tree exporter for LLM prompts.

Produces a representation of a parsed UI tree for use in LLM prompts
(state abstraction, semantic grounding, DSL guard generation, and contract
evolution). The output is semantic rather than canonical — it drops bounds,
package, and non-interactive structural wrappers — so it must NEVER be used for
fingerprinting, state identity, or action binding. Those consumers must operate
on the canonical ``UIElement`` graph.

Each emitted node gets a deterministic ``c_NNNN`` handle. These handles
are local to a single ``compact_ui_tree_text`` call and regenerate from
zero each time; do not persist them.
"""

from __future__ import annotations

from vigil.core.ui_selectors import build_component_selector
from vigil.models.state import UIElement, _normalize_dynamic

_INDENT = "    "


def _class_leaf(class_name: str | None) -> str:
    if not class_name:
        return ""
    return class_name.rsplit(".", 1)[-1]


def _short_rid(rid: str | None) -> str:
    if not rid:
        return ""
    if ":id/" in rid:
        return rid.split(":id/", 1)[1]
    return rid


def _affordances(e: UIElement) -> str:
    parts: list[str] = []
    if e.is_clickable:
        parts.append("click")
    if e.is_long_clickable:
        parts.append("long_click")
    if e.is_scrollable:
        parts.append("scroll")
    if e.is_editable:
        parts.append("input")
    if e.is_checkable:
        parts.append("check")
    return ",".join(parts)


def _status_words(e: UIElement) -> str:
    parts: list[str] = []
    if e.is_checkable and e.is_checked:
        parts.append("checked")
    if e.is_checkable and not e.is_checked:
        parts.append("unchecked")
    if e.is_selected:
        parts.append("selected")
    if not e.is_enabled:
        parts.append("disabled")
    if e.is_password:
        parts.append("password")
    return " ".join(parts)


def _selector_summary(e: UIElement, elements: list[UIElement]) -> str:
    """Compact one-token summary of the element's strongest selector signal."""
    sel = build_component_selector(e, elements)
    if sel.get("resource_id"):
        return f"rid:{_short_rid(sel['resource_id'])}"
    if sel.get("text"):
        return f"text:{sel['text'].replace(' ', '_')}"
    if sel.get("content_description"):
        return f"cd:{sel['content_description'].replace(' ', '_')}"
    if sel.get("nearby_text"):
        return f"near:{sel['nearby_text'].replace(' ', '_')}"
    cls = _class_leaf(sel.get("class_name") or "")
    return f"cls:{cls}" if cls else "cls:?"


def _is_semantic(e: UIElement) -> bool:
    """Worth emitting: interactable or carries visible text/content-desc."""
    if e.is_clickable or e.is_long_clickable or e.is_scrollable or e.is_editable or e.is_checkable:
        return True
    if (e.text or "").strip():
        return True
    return bool((e.content_description or "").strip())


def _format_node(e: UIElement, elements: list[UIElement], handle: str) -> str:
    """Build the single-line representation for one node."""
    cls = _class_leaf(e.class_name)
    rid = _short_rid(e.resource_id)
    aff = _affordances(e)
    status = _status_words(e)
    text = _normalize_dynamic(e.text) if e.text else ""
    cd = (e.content_description or "").strip()
    sel = _selector_summary(e, elements)

    parts: list[str] = [f"[{handle}]"]
    if cls:
        parts.append(cls)
    if rid:
        parts.append(rid)
    parts.append(f";{aff};")
    if status:
        parts.append(status)
    if text:
        parts.append(f'text="{text}"')
    if cd and cd != text:
        parts.append(f'cd="{cd}"')
    parts.append(f"selector={sel}")
    return " ".join(parts)


def compact_ui_tree_text(elements: list[UIElement]) -> str:
    """Render ``elements`` as an indented plain-text tree for LLM prompts.

    Walks the element tree in document order (DFS via ``children`` lists),
    and emits semantic nodes only (see ``_is_semantic``).
    Indentation is based on the depth relative to the first emitted ancestor
    so a deeply-nested but otherwise flat hierarchy doesn't waste columns.

    The output is intended for LLM consumption. It must NOT be parsed back
    into a graph and must NOT be fingerprinted — handles regenerate every
    call and the format is not stable.
    """
    if not elements:
        return ""

    by_id = {e.element_id: e for e in elements}
    # Roots = elements without a known parent in this list.
    roots = [e for e in elements if not e.parent_id or e.parent_id not in by_id]

    lines: list[str] = []
    handle_counter = [0]

    def visit(eid: str, indent_level: int) -> None:
        e = by_id.get(eid)
        if e is None:
            return
        child_indent = indent_level
        if _is_semantic(e):
            handle = f"c_{handle_counter[0]:04d}"
            handle_counter[0] += 1
            lines.append(f"{_INDENT * indent_level}{_format_node(e, elements, handle)}")
            child_indent = indent_level + 1
        for cid in e.children:
            visit(cid, child_indent)

    for root in roots:
        visit(root.element_id, 0)

    return "\n".join(lines)

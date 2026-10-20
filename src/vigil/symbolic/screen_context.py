"""Canonical conversion from runtime UI observations to DSL ``ScreenContext``."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from vigil.symbolic.dsl_evaluator import AMBIGUOUS_RESOURCE_ID_MARKER, ScreenContext

_BOOLEAN_PROPERTIES = (
    "is_clickable",
    "is_checkable",
    "is_checked",
    "is_enabled",
    "is_editable",
    "is_selected",
)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump()
    return {}


def runtime_elements_from_observation(observation: Any) -> list[dict[str, Any]]:
    """Return concrete UI elements from one runtime observation."""
    screen = _as_dict(observation)
    elements = screen.get("elements")
    interactable = screen.get("interactable_elements")
    if isinstance(elements, list) and (elements or not isinstance(interactable, list)):
        return [_as_dict(element) for element in elements]
    if isinstance(interactable, list):
        return [_as_dict(element) for element in interactable]
    return []


def _focus_subtree_ids(
    focus_element_id: str, elements_by_id: dict[str, dict[str, Any]]
) -> set[str]:
    """Element ids of ``focus_element_id`` and every descendant."""
    inside: set[str] = set()
    stack = [focus_element_id]
    while stack:
        eid = stack.pop()
        if not eid or eid in inside or eid not in elements_by_id:
            continue
        inside.add(eid)
        for child in elements_by_id[eid].get("children") or []:
            stack.append(
                child if isinstance(child, str) else str(_as_dict(child).get("element_id"))
            )
    return inside


def screen_context_from_observation(
    observation: Any, focus_element_id: str | None = None
) -> ScreenContext:
    """Build the one UI-property view used by offline replay and online verification.

    ``focus_element_id`` scopes alias resolution to one row. On a list screen the
    same ``resource_id`` is claimed by every row, so it names no single surface
    and collapses to ``AMBIGUOUS_RESOURCE_ID_MARKER`` — which makes a perfectly
    reasonable row guard such as ``read(<row title>, text) == $intent.name``
    evaluate UNKNOWN, and the verdict UNCERTAIN, even when the action is bound to
    a specific row. Given the row the action actually binds to, an id claimed by
    several elements resolves to the claimant inside that row's subtree; ids that
    already resolve uniquely across the whole screen are unaffected, so this is a
    no-op everywhere except repeated rows.
    """
    raw_elements = runtime_elements_from_observation(observation)
    elements_by_id = {
        str(element.get("element_id") or ""): element
        for element in raw_elements
        if element.get("element_id")
    }
    focus_ids = _focus_subtree_ids(focus_element_id, elements_by_id) if focus_element_id else set()
    context_elements: dict[str, dict[str, Any]] = {}

    def descendant_values(element: dict[str, Any]) -> list[dict[str, str]]:
        values: list[dict[str, str]] = []
        seen_elements: set[str] = set()
        stack = list(element.get("children") or [])
        while stack:
            child_ref = stack.pop(0)
            child = child_ref if isinstance(child_ref, dict) else elements_by_id.get(str(child_ref))
            if not isinstance(child, dict):
                continue
            child_id = str(child.get("element_id") or "")
            if child_id and child_id in seen_elements:
                continue
            if child_id:
                seen_elements.add(child_id)
            text = str(child.get("text") or child.get("content_description") or "").strip()
            if text:
                values.append({"text": text})
            stack.extend(child.get("children") or [])
        return values

    resource_owners: dict[str, str] = {}
    focused_owners: set[str] = set()
    for index, element in enumerate(raw_elements):
        text = str(element.get("text") or "")
        properties: dict[str, Any] = {
            "text": text,
            "content_description": str(element.get("content_description") or ""),
            "value": text,
            "resource_id": str(element.get("resource_id") or ""),
        }
        for key in _BOOLEAN_PROPERTIES:
            properties[key] = bool(element.get(key, key == "is_enabled"))

        raw_children = element.get("children")
        if isinstance(raw_children, list):
            properties["children"] = descendant_values(element)
            properties["children_count"] = len(raw_children)
        elif isinstance(element.get("children_count"), int):
            properties["children"] = []
            properties["children_count"] = element["children_count"]

        element_id = str(element.get("element_id") or "")
        resource_id = str(element.get("resource_id") or "")
        if element_id:
            context_elements[element_id] = properties
        if resource_id:
            # A resource_id shared by several distinct live elements (repeated
            # rows) cannot name one surface; mark it ambiguous instead of
            # letting the last element silently win. Element-id lookups stay
            # untouched — only the alias entry is replaced.
            owner = resource_owners.get(resource_id)
            claimant = element_id or f"_anon_{index}"
            in_focus = bool(focus_ids) and element_id in focus_ids
            if owner is None:
                resource_owners[resource_id] = claimant
                context_elements[resource_id] = properties
                if in_focus:
                    focused_owners.add(resource_id)
            elif owner != claimant:
                if in_focus and resource_id not in focused_owners:
                    # The focused row is the one the action binds to, so it wins the
                    # alias outright rather than colliding with its siblings.
                    resource_owners[resource_id] = claimant
                    context_elements[resource_id] = properties
                    focused_owners.add(resource_id)
                elif not in_focus and resource_id in focused_owners:
                    # A sibling row never displaces the focused row's claim.
                    continue
                else:
                    context_elements[resource_id] = {AMBIGUOUS_RESOURCE_ID_MARKER: True}

    return ScreenContext(elements=context_elements)

"""Stable widget registry for contract-first guard generation (Stage 4, step 2).

This module builds a *deterministic, LLM-free* widget registry from runtime trace
evidence. A ``WidgetRegistry`` assigns each guard-relevant element on a source screen a
stable alias (backed by ``resource_id`` / ``content_description`` / stable text role /
class alias) plus a coarse role, readable DSL property list, and runtime-readable
semantic widgets (for example amount, recipient, address, selected item, status, or
dialog text labels).

It is one of the two inputs the later typed-``GuardContract`` synthesis pass consumes
(the other being :mod:`vigil.neuro.guard_evidence`). Per project rules:

- XML/runtime traces are the source of truth: only elements actually present on the
  representative screen become registry entries.
- ``AppPrior`` (APK static artifacts) is a *prior only*. It may enrich the role of an
  already-present entry, but must never create an entry for an element that is absent
  from the runtime screen.
- Raw capture-local ``e_XXXX`` handles are never exposed as a primary alias unless no
  better signal exists; they are kept in ``element_id_to_alias`` so later stages can map
  an action target handle back to its stable alias.

No DSL compilation, admission validation, or LLM call happens here.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vigil.models.fsm import AbstractState
    from vigil.neuro.app_prior import AppPrior


class WidgetRole(StrEnum):
    """Coarse, deterministically-inferred role of a widget."""

    UNKNOWN = "unknown"
    BUTTON = "button"
    TEXT_FIELD = "text_field"
    TOGGLE = "toggle"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    LIST_CONTAINER = "list_container"
    TITLE = "title"
    IMAGE_BUTTON = "image_button"


class WidgetRegistryEntry(BaseModel):
    """A single stable widget alias and its observed/derived properties."""

    alias: str
    element_id: str = ""
    resource_id: str = ""
    text: str = ""
    content_description: str = ""
    descendant_texts: list[str] = Field(default_factory=list)
    class_name: str = ""
    role: WidgetRole = WidgetRole.UNKNOWN
    readable_props: list[str] = Field(default_factory=list)
    observed_values: dict[str, Any] = Field(default_factory=dict)
    source: str = "trace"


class WidgetRegistry(BaseModel):
    """Stable widget registry for one FSM state / source screen."""

    state_id: str
    screen_id: str | None = None
    entries: dict[str, WidgetRegistryEntry] = Field(default_factory=dict)
    element_id_to_alias: dict[str, str] = Field(default_factory=dict)
    resource_id_to_alias: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Deterministic vocabularies
# ---------------------------------------------------------------------------

# Boolean observation fields retained as evidence for source-to-target diffs.
_BOOL_PROPS: tuple[str, ...] = ("is_checked", "is_enabled")

# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _as_dict(obj: Any) -> dict[str, Any]:
    """Coerce a screen/element to a plain dict (pydantic models -> ``model_dump``)."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return {}


def _element_merge_key(el: dict[str, Any]) -> tuple[str, str]:
    """Stable de-dup key for elements that appear in both screen lists."""
    return str(el.get("element_id") or ""), str(el.get("resource_id") or "")


def _readable_descendant_texts(
    el: dict[str, Any],
    elements_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    """Return visible text/content-description values below one container.

    The traversal follows the accessibility hierarchy only.  It does not infer a
    semantic role from resource-id words or screen text.
    """
    values: list[str] = []
    seen_values: set[str] = set()
    seen_elements: set[str] = set()
    stack = list(el.get("children") or [])
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
        value = str(child.get("text") or child.get("content_description") or "").strip()
        if value and value not in seen_values:
            values.append(value)
            seen_values.add(value)
        stack.extend(child.get("children") or [])
    return values


def _has_rendered_area(el: dict[str, Any]) -> bool:
    """Reject non-action facts with explicitly empty or inverted rendered bounds.

    Accessibility text remains usable regardless of its positive pixel dimensions.
    Missing geometry leaves readability to the observed semantic content.
    """
    bounds = el.get("bounds")
    if not isinstance(bounds, list | tuple) or len(bounds) != 4:
        return True
    try:
        left, top, right, bottom = (int(value) for value in bounds)
    except (TypeError, ValueError):
        return True
    return right > left and bottom > top


def _has_readable_semantic_content(
    el: dict[str, Any],
    semantic_surface_ids: set[str],
) -> bool:
    """True when a non-action element is useful as a guard-readable source fact."""
    resource_id = str(el.get("resource_id") or "").strip()
    if not resource_id:
        return False
    if bool(el.get("is_clickable")) or bool(el.get("is_editable")) or bool(el.get("is_checkable")):
        return True
    # Non-action elements are only source facts when a user could actually read them.
    # Action widgets are exempt above: they arrive through the interactable pass and must
    # keep their aliases so action-target resolution stays stable.
    if not _has_rendered_area(el):
        return False
    if str(el.get("element_id") or "") in semantic_surface_ids:
        return True
    class_name = _short_class(str(el.get("class_name") or "")).lower()
    has_readable_value = any(str(el.get(key) or "").strip() for key in ("text", "value"))
    has_labelish_class = any(token in class_name for token in ("textview", "edittext", "button"))
    return has_readable_value and has_labelish_class


def elements_of(screen: dict[str, Any]) -> list[dict[str, Any]]:
    """Return actionable elements plus runtime-readable semantic elements."""
    ordered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    all_elements = [_as_dict(raw_el) for raw_el in screen.get("elements") or []]
    elements_by_id = {str(el.get("element_id")): el for el in all_elements if el.get("element_id")}
    semantic_surface_ids: set[str] = set()
    for action_element in all_elements:
        if not any(
            bool(action_element.get(flag))
            for flag in ("is_clickable", "is_editable", "is_checkable", "is_long_clickable")
        ):
            continue
        parent_id = str(action_element.get("parent_id") or "")
        seen_parents: set[str] = set()
        while parent_id and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent = elements_by_id.get(parent_id)
            if parent is None:
                break
            if str(parent.get("resource_id") or "").strip() and _readable_descendant_texts(
                parent,
                elements_by_id,
            ):
                semantic_surface_ids.add(parent_id)
                break
            parent_id = str(parent.get("parent_id") or "")

    def add(raw_elements: Any, *, readable_filter: bool) -> None:
        for raw_el in raw_elements or []:
            el = _as_dict(raw_el)
            if readable_filter and not _has_readable_semantic_content(
                el,
                semantic_surface_ids,
            ):
                continue
            key = _element_merge_key(el)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(el)

    # Keep action widgets first so action-target aliasing remains stable.
    add(screen.get("interactable_elements"), readable_filter=False)
    add(screen.get("elements"), readable_filter=True)
    return ordered


def _short_class(class_name: str) -> str:
    return class_name.rsplit(".", 1)[-1] if "." in class_name else class_name


def _slug(value: str) -> str:
    """Lowercase, non-alphanumeric -> ``_``, collapse and trim underscores."""
    out: list[str] = []
    prev_us = False
    for ch in value.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        elif not prev_us:
            out.append("_")
            prev_us = True
    return "".join(out).strip("_")


def _resource_short(resource_id: str) -> str:
    """Short name from a resource id, e.g. ``com.app:id/amount`` -> ``amount``."""
    short = resource_id
    if ":id/" in short:
        short = short.split(":id/", 1)[-1]
    elif "/" in short:
        short = short.rsplit("/", 1)[-1]
    return _slug(short)


# ---------------------------------------------------------------------------
# Per-element inference
# ---------------------------------------------------------------------------


def _infer_role(el: dict[str, Any]) -> WidgetRole:
    cls = _short_class(str(el.get("class_name") or "")).lower()
    rid = str(el.get("resource_id") or "").lower()
    text = str(el.get("text") or "").lower()

    if "edittext" in cls or el.get("is_editable"):
        return WidgetRole.TEXT_FIELD
    if "switch" in cls:
        return WidgetRole.TOGGLE
    if "checkbox" in cls:
        return WidgetRole.CHECKBOX
    if "radiobutton" in cls:
        return WidgetRole.RADIO
    if "imagebutton" in cls:
        return WidgetRole.IMAGE_BUTTON
    if "button" in cls:
        return WidgetRole.BUTTON
    if any(k in cls for k in ("recyclerview", "listview", "scrollview")) or el.get("is_scrollable"):
        return WidgetRole.LIST_CONTAINER
    if "title" in rid or "title" in text:
        return WidgetRole.TITLE
    if el.get("is_checkable"):
        return WidgetRole.CHECKBOX
    return WidgetRole.UNKNOWN


def _readable_props(
    el: dict[str, Any],
    role: WidgetRole,
    descendant_texts: list[str],
) -> list[str]:
    props: list[str] = []
    # text / content_description / value are exposed by the runtime screen context
    # (DecisionEngine._build_screen_context) even when empty — an empty EditText / value
    # is meaningful for form guards. Include them whenever the key is present.
    for key in ("text", "content_description", "value"):
        if key in el:
            props.append(key)
    if descendant_texts and "value" not in props:
        props.append("value")
    for name in ("is_checked", "is_selected"):
        if name in el:
            props.append(name)
    if role is WidgetRole.LIST_CONTAINER and ("children" in el or "children_count" in el):
        props.append("children_count")
    return props


def _observed_values(el: dict[str, Any]) -> dict[str, Any]:
    """Preserve trace-observed boolean/count facts used by evidence diffs."""
    values = {name: el[name] for name in _BOOL_PROPS if name in el}
    if "children_count" in el:
        values["children_count"] = el["children_count"]
    elif "children" in el and isinstance(el["children"], list):
        values["children_count"] = len(el["children"])
    return values


def _alias_candidate(el: dict[str, Any], role: WidgetRole) -> str:
    """Best stable alias signal for an element, ``""`` if only the raw id is available."""
    resource_id = str(el.get("resource_id") or "").strip()
    if resource_id:
        short = _resource_short(resource_id)
        if short:
            return short
    cdesc = _slug(str(el.get("content_description") or ""))
    if cdesc:
        return cdesc
    text = _slug(str(el.get("text") or ""))
    if text:
        return f"{text}_{role.value}"
    return ""


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_widget_registry_from_screen(
    state_id: str,
    screen: dict[str, Any],
    app_prior: AppPrior | None = None,
) -> WidgetRegistry:
    """Build a :class:`WidgetRegistry` from a single representative screen dict."""
    registry = WidgetRegistry(
        state_id=state_id,
        screen_id=screen.get("screen_id"),
    )

    used_aliases: set[str] = set()
    class_counts: dict[str, int] = {}
    ambiguous_resource_ids: set[str] = set()
    all_elements = [_as_dict(raw_el) for raw_el in screen.get("elements") or []]
    elements_by_id = {str(el.get("element_id")): el for el in all_elements if el.get("element_id")}

    for el in elements_of(screen):
        element_id = str(el.get("element_id") or "")
        resource_id = str(el.get("resource_id") or "")
        role = _infer_role(el)
        descendant_texts = _readable_descendant_texts(el, elements_by_id)

        alias = _alias_candidate(el, role)
        if not alias:
            short_class = _short_class(str(el.get("class_name") or ""))
            if short_class:
                idx = class_counts.get(short_class, 0)
                class_counts[short_class] = idx + 1
                alias = f"{short_class}_{idx}"
            else:
                alias = element_id or f"element_{len(registry.entries)}"

        # Ensure uniqueness within this registry.
        base_alias = alias
        suffix = 2
        while alias in used_aliases:
            alias = f"{base_alias}_{suffix}"
            suffix += 1
        used_aliases.add(alias)

        entry = WidgetRegistryEntry(
            alias=alias,
            element_id=element_id,
            resource_id=resource_id,
            text=str(el.get("text") or ""),
            content_description=str(el.get("content_description") or ""),
            descendant_texts=descendant_texts,
            class_name=str(el.get("class_name") or ""),
            role=role,
            readable_props=_readable_props(el, role, descendant_texts),
            observed_values=_observed_values(el),
        )
        registry.entries[alias] = entry
        if element_id:
            registry.element_id_to_alias[element_id] = alias
        if resource_id:
            if resource_id in ambiguous_resource_ids:
                continue
            if resource_id in registry.resource_id_to_alias:
                registry.resource_id_to_alias.pop(resource_id)
                ambiguous_resource_ids.add(resource_id)
            else:
                registry.resource_id_to_alias[resource_id] = alias

    if app_prior is not None:
        _enrich_with_prior(registry, app_prior)

    return registry


def build_widget_registry(
    state: AbstractState,
    raw_screens: dict[str, Any],
    app_prior: AppPrior | None = None,
) -> WidgetRegistry:
    """Build a :class:`WidgetRegistry` for ``state`` from its representative screen.

    Uses the first id in ``state.evidence.raw_screen_ids`` that resolves in
    ``raw_screens``. If the state has no usable screen, returns an empty registry that
    still carries the ``state_id`` (no crash).
    """
    screen: dict[str, Any] | None = None
    screen_id: str | None = None
    for sid in state.evidence.raw_screen_ids:
        candidate = raw_screens.get(sid)
        if candidate is not None:
            screen = _as_dict(candidate)
            screen_id = sid
            break

    if screen is None:
        return WidgetRegistry(state_id=state.state_id)

    registry = build_widget_registry_from_screen(state.state_id, screen, app_prior)
    # Prefer the screen-id key we matched in raw_screens when the screen dict omits it.
    if registry.screen_id is None:
        registry.screen_id = screen_id
    return registry


def build_widget_registry_from_screen_ids(
    state_id: str,
    screen_ids: list[str],
    raw_screens: dict[str, Any],
    app_prior: AppPrior | None = None,
) -> WidgetRegistry | None:
    """Build a registry from the first id in ``screen_ids`` that resolves.

    Returns ``None`` when none of the ids resolves in ``raw_screens`` (so the caller can
    fall back to a state-evidence-based screen). Used to prefer per-transition
    provenance screens, which deterministically contain the capture-local action target
    handle, over a merged/quotiented state's first representative screen.
    """
    for sid in screen_ids:
        candidate = raw_screens.get(sid)
        if candidate is not None:
            registry = build_widget_registry_from_screen(state_id, _as_dict(candidate), app_prior)
            if registry.screen_id is None:
                registry.screen_id = sid
            return registry
    return None


def _enrich_with_prior(registry: WidgetRegistry, app_prior: AppPrior) -> None:
    """Enrich *existing* entries with static-prior hints. Never creates new entries.

    Static APK artifacts are priors only: we may strengthen widget role metadata or mark
    a static-prior corroboration, but element presence still comes from the runtime
    screen.
    """
    # Widget declarations indexed by short id for cheap lookup.
    decl_classes: dict[str, str] = {}
    for decl in app_prior.widget_declarations:
        decl_classes[_slug(decl.widget_id)] = decl.widget_class

    for entry in registry.entries.values():
        if entry.resource_id:
            short = _resource_short(entry.resource_id)
            decl_class = decl_classes.get(short)
            if decl_class and entry.role is WidgetRole.UNKNOWN:
                inferred = _infer_role({"class_name": decl_class})
                if inferred is not WidgetRole.UNKNOWN:
                    entry.role = inferred
                    entry.source = "trace+prior"

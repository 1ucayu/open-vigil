"""Accessibility tree XML parser.

Parses Android accessibility tree XML (from uiautomator2 dump_hierarchy()) into
structured UIElement representations. Reference: V-Droid html_representation.py
for element filtering and display_id assignment patterns.
"""

from __future__ import annotations

import re
from xml.etree import ElementTree as ET

from loguru import logger

from vigil.models.state import UIElement

# System packages whose elements should be excluded from parsing.
# These are Android system chrome (status bar, navigation bar) — not part
# of the target app and they cause fingerprint noise + wasted actions.
SYSTEM_PACKAGES: set[str] = {
    "com.android.systemui",
}

SYSTEM_CHROME_RESOURCE_IDS: set[str] = {
    "android:id/statusBarBackground",
}


def parse_hierarchy_xml(
    xml_string: str,
    app_package: str | None = None,
) -> list[UIElement]:
    """Parse uiautomator2 dump_hierarchy() XML into a flat list of UIElements.

    Args:
        xml_string: Raw XML string from device.dump_hierarchy().
        app_package: If provided, filter out elements from system packages
            (status bar, nav bar) to reduce noise. Elements from the target
            app's package and non-system packages are kept.

    Returns:
        Flat list of UIElement objects with unique element_ids assigned
        via DFS traversal order. Returns empty list on parse failure.
    """
    if not xml_string or not xml_string.strip():
        return []

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        logger.warning("Failed to parse accessibility tree XML")
        return []

    elements: list[UIElement] = []
    counter = [0]  # mutable counter for sequential ID assignment

    for child in root:
        if child.tag == "node":
            _parse_node(
                child,
                depth=0,
                counter=counter,
                elements=elements,
                app_package=app_package,
            )

    logger.debug(f"Parsed {len(elements)} UI elements from hierarchy XML")
    return elements


def parse_bounds(bounds_str: str) -> list[int]:
    """Parse bounds string '[left,top][right,bottom]' into [left, top, right, bottom].

    Args:
        bounds_str: Bounds string from uiautomator2 XML, e.g. "[100,200][300,400]".

    Returns:
        List of 4 integers [left, top, right, bottom]. Returns [0, 0, 0, 0] on failure.
    """
    matches = re.findall(r"\d+", bounds_str)
    if len(matches) == 4:
        return [int(x) for x in matches]
    return [0, 0, 0, 0]


def _parse_node(
    node: ET.Element,
    depth: int,
    counter: list[int],
    elements: list[UIElement],
    app_package: str | None = None,
    parent_id: str | None = None,
) -> str | None:
    """Recursively parse a <node> element into UIElement(s).

    Args:
        node: XML Element to parse.
        depth: Current depth in the tree.
        counter: Mutable counter for sequential element_id assignment.
        elements: Accumulator list for parsed elements.
        app_package: If set, skip subtrees belonging to SYSTEM_PACKAGES.
        parent_id: Element ID of the parent node.

    Returns:
        The element_id of the parsed node, or None if the node was filtered.
    """
    if app_package:
        pkg = node.attrib.get("package", "")
        if pkg in SYSTEM_PACKAGES:
            return None
        if node.attrib.get("resource-id") in SYSTEM_CHROME_RESOURCE_IDS:
            return None

    element_id = f"e_{counter[0]:04d}"
    counter[0] += 1

    child_ids = []
    for child in node:
        if child.tag == "node":
            child_id = _parse_node(
                child, depth + 1, counter, elements, app_package, parent_id=element_id
            )
            if child_id is not None:
                child_ids.append(child_id)

    def _bool(attr_name: str) -> bool:
        return node.attrib.get(attr_name, "false") == "true"

    class_name = node.attrib.get("class", "")
    node_package = node.attrib.get("package", "")
    is_editable = _bool("focusable") and class_name.endswith("EditText")

    element = UIElement(
        element_id=element_id,
        class_name=class_name,
        package=node_package,
        resource_id=node.attrib.get("resource-id") or None,
        text=node.attrib.get("text") or None,
        content_description=node.attrib.get("content-desc") or None,
        bounds=parse_bounds(node.attrib.get("bounds", "[0,0][0,0]")),
        is_clickable=_bool("clickable"),
        is_long_clickable=_bool("long-clickable"),
        is_scrollable=_bool("scrollable"),
        is_editable=is_editable,
        is_checkable=_bool("checkable"),
        is_checked=_bool("checked"),
        is_enabled=_bool("enabled"),
        is_selected=_bool("selected"),
        is_password=_bool("password"),
        depth=depth,
        children=child_ids,
        parent_id=parent_id,
    )
    elements.append(element)
    return element_id

"""Convert runtime UI-hierarchy XML into a RawScreen.

Reuses :func:`vigil.core.ui_parser.parse_hierarchy_xml`, the same deterministic
parser used offline, so runtime fingerprinting, selectors, and replay stay on a
single source of truth. Nothing here requires a live device.
"""

from __future__ import annotations

from vigil.core.ui_parser import parse_hierarchy_xml
from vigil.models.state import RawScreen


def raw_screen_from_xml(
    xml_text: str,
    screen_id: str = "runtime",
    app_package: str | None = None,
    activity_name: str | None = None,
    package_name: str | None = None,
) -> RawScreen:
    """Parse a UI-hierarchy XML dump into a :class:`RawScreen`.

    Args:
        xml_text: Raw accessibility-hierarchy XML.
        screen_id: Identifier for the produced screen.
        app_package: Package used only to filter system-package noise during
            parsing (see ``parse_hierarchy_xml``). Not stored on the screen.
        activity_name: Foreground activity, when known. Not invented.
        package_name: Foreground package, when known. Not invented.

    Returns:
        A ``RawScreen`` with parsed elements. Only supplied context fields are
        populated — the helper never fabricates activity/package metadata.
    """
    elements = parse_hierarchy_xml(xml_text, app_package)
    return RawScreen(
        screen_id=screen_id,
        activity_name=activity_name,
        package_name=package_name,
        elements=elements,
    )

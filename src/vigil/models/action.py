"""Action type definitions and templates for UI interaction.

Defines the vocabulary of actions that Vigil's explorer can perform on UI elements
and that the FSM records as transition labels.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(StrEnum):
    """Types of UI actions."""

    CLICK = "click"
    LONG_PRESS = "long_press"
    INPUT_TEXT = "input_text"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    NAVIGATE_BACK = "navigate_back"
    NAVIGATE_HOME = "navigate_home"


class Action(BaseModel):
    """A raw action ``a_t`` proposed or executed on observation ``o_t``.

    Canonicalization derives the canonical action ``sigma_t`` from ``action_type``
    and a stable selector using the source observation. ``target_element_id`` and
    ``target_bounds`` are capture-local hints and do not identify a transition.
    Concrete input values remain in the raw action and are checked by guardrails.

    Attributes:
        action_type: The type of action to perform.
        target_resource_id: Android resource id of the target element.
        target_text: Displayed text of the target (normalized at build time).
        target_content_desc: Accessibility content-description of the target.
        target_class_name: Android widget class name (last-resort discriminator).
        target_element_id: Capture-local id (volatile hint).
        target_bounds: Capture-local bounds (volatile hint; re-resolved at exec).
        target_selector: Stable component selector (see vigil.core.ui_selectors).
        input_text: Text to input (only for INPUT_TEXT actions).
        metadata: Arbitrary action metadata.
    """

    action_type: ActionType
    target_element_id: str | None = None
    target_bounds: list[int] | None = None
    target_resource_id: str | None = None
    target_text: str | None = None
    target_content_desc: str | None = None
    target_class_name: str | None = None
    target_selector: dict[str, Any] = Field(default_factory=dict)
    input_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_fsm_dict(self) -> dict[str, Any]:
        """Convert to the trace action sample format before FSM canonicalization."""
        result: dict[str, Any] = {"type": self.action_type.value}
        if self.target_element_id:
            result["target"] = self.target_element_id
        if self.target_bounds:
            result["bounds"] = self.target_bounds
        if self.target_resource_id:
            result["resource_id"] = self.target_resource_id
            result["target_resource_id"] = self.target_resource_id
        if self.target_text:
            result["target_text"] = self.target_text
        if self.target_content_desc:
            result["target_content_desc"] = self.target_content_desc
        if self.target_class_name:
            result["target_class_name"] = self.target_class_name
        if self.target_selector:
            result["target_selector"] = self.target_selector
        if self.input_text:
            result["text"] = self.input_text
        return result

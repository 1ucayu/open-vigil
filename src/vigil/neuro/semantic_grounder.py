"""Stage 2.5: Semantic Grounding.

Enriches abstract-state annotations with LLM-generated state descriptions. Invariant
generation and container abstraction are owned by their dedicated construction pipelines
and are not modified here.

All LLM calls are multimodal (accessibility tree + screenshot). Activity priors from
Stage 0 provide top-down context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from vigil.core.llm_client import LlmClient
from vigil.models.fsm import (
    AbstractState,
    AppFSM,
    StateAnnotations,
)
from vigil.neuro.app_prior import AppPrior

_DESC_SYSTEM_PROMPT = """\
You are annotating mobile app UI states for a runtime verification system.
You will receive a screenshot and an accessibility tree of one screen.
Return ONLY valid JSON with these fields:
{
  "alt_text": "description a blind user would understand",
  "page_function": "hierarchical/slash/path (e.g. settings/wifi/list)"
}
Do NOT hallucinate elements not visible in the screenshot.
Use a consistent hierarchical vocabulary for page_function."""


def _match_layout_to_activity(layout_file: str, activity_name: str | None) -> bool:
    """Heuristically match a layout file name to an Activity name."""
    if not activity_name:
        return False
    short = activity_name.rsplit(".", 1)[-1]
    short = short.replace("Activity", "").replace("Fragment", "").lower()
    return short in layout_file.lower()


def _build_static_context(
    app_prior: AppPrior | None,
    activity_name: str | None,
    elements: list[dict[str, Any]],
) -> str:
    """Build static context string from AppPrior for LLM prompt injection."""
    if not app_prior:
        return ""

    parts: list[str] = []

    if app_prior.widget_declarations:
        relevant = [
            w
            for w in app_prior.widget_declarations
            if _match_layout_to_activity(w.layout_file, activity_name)
        ]
        if not relevant:
            relevant = list(app_prior.widget_declarations)
        wl = ", ".join(f"{w.widget_class}(id={w.widget_id})" for w in relevant)
        parts.append(f"Declared widgets in layout XML: {wl}")

    if app_prior.string_constants and elements:
        visible_texts = {e.get("text", "") for e in elements if e.get("text")}
        matched = [
            f"{k}={v!r}" for k, v in app_prior.string_constants.items() if v in visible_texts
        ]
        if matched:
            parts.append(f"Matched string constants: {', '.join(matched)}")
        else:
            constants = [f"{k}={v!r}" for k, v in app_prior.string_constants.items()]
            parts.append(f"String constants: {', '.join(constants)}")

    if app_prior.string_arrays:
        summaries: list[str] = []
        for name, values in app_prior.string_arrays.items():
            summaries.append(f"{name}: [{', '.join(str(value) for value in values)}]")
        parts.append(f"String arrays: {'; '.join(summaries)}")

    if app_prior.permissions:
        permissions = [p.replace("android.permission.", "") for p in app_prior.permissions]
        parts.append(f"App permissions: {', '.join(permissions)}")

    if not parts:
        return ""
    return "## Static app context (from APK)\n" + "\n".join(parts) + "\n\n"


class SemanticGrounder:
    """Stage 2.5: Enrich FSM states with semantic metadata via multimodal LLM.

    Args:
        llm: Configured LLM client for multimodal generation.
    """

    def __init__(self, llm: LlmClient) -> None:
        self._llm = llm

    def ground_all_states(
        self,
        fsm: AppFSM,
        raw_screens: dict[str, dict[str, Any]],
        app_prior: AppPrior | None = None,
    ) -> AppFSM:
        """Run all grounding sub-tasks on every state in the FSM.

        Args:
            fsm: The app FSM to enrich (modified in place and returned).
            raw_screens: Map of screen_id -> screen data dict (from trace JSON).
                Each entry should have at least 'screenshot_path', 'xml_tree_path',
                and 'interactable_elements'.
            app_prior: Optional app prior from Stage 0 for cross-validation.
        Returns:
            The same FSM with enriched state metadata.
        """
        for state_id, state in fsm.states.items():
            logger.info(f"Grounding state {state_id} ({state.name})")

            observations = self._collect_observations(state, raw_screens)
            if not observations:
                logger.warning(f"No observations for state {state_id}, skipping")
                continue

            try:
                annotations = self.generate_state_description(state, observations, app_prior)
            except ValueError as exc:
                logger.warning(f"State description rejected for {state_id}: {exc}")
                continue
            annotations.display_name = state.annotations.display_name
            state.annotations = annotations

        return fsm

    def generate_state_description(
        self,
        state: AbstractState,
        observations: list[dict[str, Any]],
        app_prior: AppPrior | None = None,
    ) -> StateAnnotations:
        """Generate semantic description for a state via multimodal LLM.

        Raises:
            ValueError: The model response cannot be parsed as a JSON description.
        """
        obs = observations[0]
        prompt_parts: list[str] = []

        if app_prior and state.android_context.activity_name:
            activity_info = _find_activity(app_prior, state.android_context.activity_name)
            if activity_info:
                prompt_parts.append(f"This screen belongs to Activity: {activity_info.name}")
                if activity_info.label:
                    prompt_parts.append(f"Activity label: {activity_info.label}")

        elements_raw = obs.get("interactable_elements", obs.get("elements", []))
        static_ctx = _build_static_context(
            app_prior, state.android_context.activity_name, elements_raw
        )
        if static_ctx:
            prompt_parts.append(static_ctx)

        element_table = _build_element_table(obs)
        prompt_parts.append(f"Accessibility tree elements:\n{element_table}")
        prompt_parts.append(
            "Generate a JSON annotation for this screen with: alt_text, page_function."
        )
        user_prompt = "\n\n".join(prompt_parts)

        screenshot_path = obs.get("screenshot_path")
        if screenshot_path and Path(screenshot_path).exists():
            response = self._llm.generate_with_images(
                _DESC_SYSTEM_PROMPT,
                user_prompt,
                [Path(screenshot_path)],
                ["Current screen:"],
            )
        else:
            response = self._llm.generate(_DESC_SYSTEM_PROMPT, user_prompt)

        parsed = _parse_json(response)
        if parsed is None:
            raise ValueError(f"Invalid state description JSON for {state.state_id}")

        return StateAnnotations(
            alt_text=str(parsed.get("alt_text", "")),
            page_function=str(parsed.get("page_function", "")),
        )

    @staticmethod
    def _collect_observations(
        state: AbstractState,
        raw_screens: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Collect all observation dicts for a state's raw screens."""
        return [raw_screens[sid] for sid in state.evidence.raw_screen_ids if sid in raw_screens]


def _find_activity(prior: AppPrior, activity_name: str) -> Any | None:
    """Find an ActivityInfo by name in the prior."""
    for a in prior.activities:
        if a.name == activity_name:
            return a
    return None


def _build_element_table(obs: dict[str, Any]) -> str:
    """Build an element summary table from an observation dict."""
    elements = obs.get("interactable_elements", [])
    if not elements:
        return "(no interactable elements)"

    lines: list[str] = []
    for e in elements:
        eid = e.get("element_id", "?")
        cls = (e.get("class_name") or "").rsplit(".", 1)[-1]
        text = e.get("text") or ""
        rid = e.get("resource_id") or ""
        desc = e.get("content_description") or ""
        parts = [f"{cls}[{eid}]"]
        if rid:
            parts.append(f"rid={rid}")
        if text:
            parts.append(f'text="{text}"')
        if desc:
            parts.append(f'desc="{desc}"')
        flags: list[str] = []
        if e.get("is_clickable"):
            flags.append("click")
        if e.get("is_scrollable"):
            flags.append("scroll")
        if e.get("is_checkable"):
            flags.append(f"check={'✓' if e.get('is_checked') else '✗'}")
        if flags:
            parts.append(f"[{','.join(flags)}]")
        lines.append("  " + " ".join(parts))
    return "\n".join(lines)


def _parse_json(response: str) -> Any | None:
    """Parse LLM response as JSON, stripping markdown fences if present."""
    text = response.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except ValueError:
        return None

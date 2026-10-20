"""Focused screenshot/layout grounding for guard generation.

This module is a lightweight alternative to :mod:`vigil.neuro.semantic_grounder`.
It only enriches state annotations with a visual alt-text caption; it does not
mine invariants, alter state identity, add transitions, or make runtime decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from vigil.core.llm_client import LlmClient
from vigil.models.fsm import AppFSM, StateAnnotations
from vigil.neuro.app_prior import AppPrior

_VISUAL_SYSTEM_PROMPT = """\
You are annotating Android mobile GUI screenshots for a runtime verifier.
You receive one screenshot plus a compact accessibility element table.

Return ONLY a concise plain-text visual caption. Do not return JSON, markdown,
tables, bullets, or code fences.

Rules:
- Describe only screenshot-visible facts that are missing from or ambiguous in
  the accessibility/XML table.
- Focus on icon semantics, visual grouping, selected/disabled/occluded visual
  state, dialogs/overlays, and layout relationships.
- Refer to widgets by existing element ids, resource ids, visible text, or
  content descriptions when available.
- Do not restate ordinary XML-visible text/resource ids unless needed to explain
  a visual relationship.
- Do not invent transitions, guards, state ids, user intent values, or runtime verdicts.
- Keep the caption stable and app-agnostic. Avoid benchmark-specific labels,
  package names, local paths, or hidden evaluator facts.
"""


def ground_fsm_visual_annotations(
    fsm: AppFSM,
    raw_screens: dict[str, dict[str, Any]],
    llm: LlmClient,
    app_prior: AppPrior | None = None,
    *,
    force: bool = False,
    max_states: int | None = None,
) -> list[dict[str, Any]]:
    """Annotate FSM states with LLM-derived visual layout and icon-label metadata.

    The FSM is modified in place. Returned rows are intended for the audit report the
    caller writes beside the enriched bundle.
    """
    report: list[dict[str, Any]] = []
    attempted = 0

    for state_id, state in fsm.states.items():
        if max_states is not None and attempted >= max_states:
            break

        if not force and state.annotations.alt_text:
            report.append(
                {
                    "state_id": state_id,
                    "status": "skipped_existing",
                    "screen_id": None,
                    "page_function": state.annotations.page_function,
                    "alt_text_chars": len(state.annotations.alt_text),
                }
            )
            continue

        attempted += 1
        obs = _first_observation(state.evidence.raw_screen_ids, raw_screens)
        if obs is None:
            report.append(
                {
                    "state_id": state_id,
                    "status": "skipped_no_observation",
                    "screen_id": None,
                }
            )
            continue

        screen_id = str(obs.get("screen_id") or state.evidence.raw_screen_ids[0])
        try:
            parsed = describe_screen_visuals(
                state_id=state_id,
                observation=obs,
                llm=llm,
                app_prior=app_prior,
            )
        except Exception as exc:
            logger.warning(f"Visual grounding failed for {state_id}: {exc}")
            report.append(
                {
                    "state_id": state_id,
                    "status": "failed",
                    "screen_id": screen_id,
                    "error": str(exc),
                }
            )
            continue

        alt_text = parsed["alt_text"]
        _apply_visual_annotation(fsm, state_id, alt_text)
        report.append(
            {
                "state_id": state_id,
                "status": "annotated",
                "screen_id": screen_id,
                "alt_text_chars": len(alt_text),
            }
        )

    return report


def describe_screen_visuals(
    *,
    state_id: str,
    observation: dict[str, Any],
    llm: LlmClient,
    app_prior: AppPrior | None = None,
) -> dict[str, str]:
    """Ask the LLM for a visual/layout description of one observed screen."""
    prompt = _build_visual_prompt(state_id, observation, app_prior)
    image_path = _existing_screenshot_path(observation)
    if image_path is not None:
        response = llm.generate_with_images(
            _VISUAL_SYSTEM_PROMPT,
            prompt,
            [image_path],
            [f"state={state_id} screen={observation.get('screen_id', '')}"],
        )
    else:
        response = llm.generate(_VISUAL_SYSTEM_PROMPT, prompt)

    alt_text = _clean_caption(response)
    if not alt_text:
        raise ValueError("visual grounding response was empty")
    return {"alt_text": alt_text}


def write_visual_grounding_report(report: list[dict[str, Any]], path: Path) -> None:
    """Write a visual grounding audit report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _first_observation(
    screen_ids: list[str],
    raw_screens: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for sid in screen_ids:
        obs = raw_screens.get(sid)
        if obs is not None:
            return obs
    return None


def _apply_visual_annotation(fsm: AppFSM, state_id: str, alt_text: str) -> None:
    state = fsm.states[state_id]
    current = state.annotations

    state.annotations = StateAnnotations(
        display_name=current.display_name,
        alt_text=alt_text,
        page_function=current.page_function,
    )


def _build_visual_prompt(
    state_id: str,
    observation: dict[str, Any],
    app_prior: AppPrior | None,
) -> str:
    elements = observation.get("interactable_elements") or observation.get("elements") or []
    parts = [
        f"state_id: {state_id}",
        f"screen_id: {observation.get('screen_id', '')}",
        f"activity: {observation.get('activity_name', '')}",
        f"page_title: {observation.get('page_title', '')}",
        "Focus on layout, visible grouping, forms/lists/dialogs, and functional icon labels.",
    ]

    if app_prior is not None:
        parts.append(_static_prior_summary(app_prior))

    parts.append("Interactable/accessibility elements:")
    parts.extend(_element_lines(elements))
    return "\n".join(part for part in parts if part)


def _element_lines(elements: list[dict[str, Any]]) -> list[str]:
    if not elements:
        return ["  (no elements)"]

    lines: list[str] = []
    for element in elements:
        eid = str(element.get("element_id") or "?")
        cls = str(element.get("class_name") or "").rsplit(".", 1)[-1]
        rid = str(element.get("resource_id") or "")
        text = str(element.get("text") or "")
        desc = str(element.get("content_description") or "")
        bounds = element.get("bounds")
        flags: list[str] = []
        for key, name in (
            ("is_clickable", "click"),
            ("is_editable", "edit"),
            ("is_checkable", "checkable"),
            ("is_checked", "checked"),
            ("is_scrollable", "scroll"),
            ("is_enabled", "enabled"),
        ):
            if element.get(key):
                flags.append(name)

        pieces = [f"  - {eid}", f"class={cls}"]
        if rid:
            pieces.append(f"rid={rid}")
        if text:
            pieces.append(f"text={text!r}")
        if desc:
            pieces.append(f"desc={desc!r}")
        if bounds:
            pieces.append(f"bounds={bounds}")
        if flags:
            pieces.append(f"flags={','.join(flags)}")
        lines.append(" ".join(pieces))
    return lines


def _static_prior_summary(app_prior: AppPrior) -> str:
    parts = [f"app_prior_package: {app_prior.package_name}"]
    if app_prior.activities:
        labels = [
            f"{activity.name} label={activity.label or ''}" for activity in app_prior.activities
        ]
        parts.append("activities: " + "; ".join(labels))
    if app_prior.permissions:
        parts.append("permissions: " + ", ".join(app_prior.permissions))
    if app_prior.string_arrays:
        names = list(app_prior.string_arrays)
        parts.append("string_arrays: " + ", ".join(names))
    return "\n".join(parts)


def _existing_screenshot_path(observation: dict[str, Any]) -> Path | None:
    raw_path = observation.get("screenshot_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.exists():
        return path
    return None


def _clean_caption(response: str) -> str:
    """Normalize a plain-text visual caption while tolerating fenced model output."""
    text = response.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text

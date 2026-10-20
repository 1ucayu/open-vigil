"""System-prompt/spec storage and a tiny loader.

Prompts for offline LLM stages live here as spec files so they can be edited and
debugged without touching Python. Load one by file name:

    from vigil.system_prompt import load_system_prompt
    text = load_system_prompt("transition_guard_generation.spec")

Paths are resolved relative to this package directory, so the same call works in an
editable ``uv`` checkout and in a built wheel (the prompt/spec files ship inside the
``vigil.system_prompt`` package).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["load_system_prompt", "PROMPT_DIR"]

PROMPT_DIR = Path(__file__).resolve().parent


def load_system_prompt(name: str) -> str:
    """Return the text of the prompt file ``name`` under ``src/vigil/system_prompt/``.

    ``name`` is a bare file name (e.g. ``"transition_guard_generation.spec"``); path
    separators are rejected so callers cannot escape the prompt directory. Raises
    :class:`FileNotFoundError` with a clear message when the prompt is missing.
    """
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"invalid prompt name: {name!r}")
    path = PROMPT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"system prompt {name!r} not found at {path} "
            f"(available: {sorted(p.name for p in PROMPT_DIR.iterdir() if p.is_file())})"
        )
    return path.read_text(encoding="utf-8")

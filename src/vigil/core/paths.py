"""Shared project path helpers."""

from __future__ import annotations

from pathlib import Path

PACKAGED_DSL_GRAMMAR = Path(__file__).resolve().parents[1] / "symbolic" / "dsl_grammar.lark"


def resolve_dsl_grammar_path(grammar_path: str | Path | None = None) -> Path:
    """Return an explicit grammar path or Vigil's packaged canonical grammar."""
    if grammar_path is not None:
        return Path(grammar_path)
    return PACKAGED_DSL_GRAMMAR

"""Run-local evidence store for observed-transition evolution.

Bridge calls may use separate processes, so an evolution engine's in-memory
``raw_screens`` may not survive to the next call. Provenance
``raw_screen_id``s written into ``fsm.json`` (state evidence and transition provenance)
must stay resolvable across later calls — otherwise a subsequent abstraction refinement
cannot reconstruct the observations behind an existing state. This module persists the
complete :class:`RawScreen` for every executed pre/post capture under

    ``$VIGIL_RUN_ARTIFACTS_DIR/runtime_evidence/<screen_id>.json``

and rehydrates them (merged with the immutable exploration trace) into real
``RawScreen`` objects before an engine is constructed. Screen ids are allocated
uniquely by the engine, so stores are merged by plain id. The original exploration
trace file is never modified.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vigil.models.state import RawScreen

_RUN_ARTIFACTS_ENV = "VIGIL_RUN_ARTIFACTS_DIR"
_EVIDENCE_SUBDIR = "runtime_evidence"


def runtime_evidence_dir() -> Path:
    """Return ``$VIGIL_RUN_ARTIFACTS_DIR/runtime_evidence`` (fails closed if unset)."""
    root = os.environ.get(_RUN_ARTIFACTS_ENV)
    if not root:
        raise ValueError(f"runtime evidence requires {_RUN_ARTIFACTS_ENV}")
    return Path(root) / _EVIDENCE_SUBDIR


def persist_runtime_screen(screen: RawScreen) -> Path:
    """Persist one executed capture as complete RawScreen JSON.

    The filename is the (unique) engine-allocated ``screen_id``.
    """
    directory = runtime_evidence_dir()
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"{screen.screen_id}.json"
    payload = json.dumps(screen.model_dump(mode="json"), indent=2, sort_keys=True)
    dest.write_text(payload, encoding="utf-8")
    return dest


def normalize_trace_screen(value: RawScreen | dict[str, Any]) -> RawScreen:
    """Coerce one raw-screen record (trace dict or model) into a ``RawScreen``."""
    return RawScreen.model_validate(value)


def load_persisted_runtime_screens() -> dict[str, RawScreen]:
    """Rehydrate every persisted runtime capture (missing dir → empty)."""
    directory = runtime_evidence_dir()
    if not directory.is_dir():
        return {}
    screens: dict[str, RawScreen] = {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        screen = normalize_trace_screen(data)
        screens[screen.screen_id] = screen
    return screens


def merge_evidence(*sources: dict[str, RawScreen]) -> dict[str, RawScreen]:
    """Merge raw-screen stores keyed by their unique engine-allocated ids."""
    merged: dict[str, RawScreen] = {}
    for source in sources:
        merged.update(source)
    return merged


def load_runtime_evidence(
    trace_screens: dict[str, RawScreen | dict[str, Any]] | None,
) -> dict[str, RawScreen]:
    """Hydrate the full raw-screen store: exploration trace + persisted runtime captures.

    ``trace_screens`` is the normalized exploration trace (e.g. the dicts returned by the
    adapter's trace loader). Both sources are coerced to ``RawScreen`` and merged, keyed
    by the embedded ``screen_id``.
    """
    trace_store: dict[str, RawScreen] = {}
    for value in (trace_screens or {}).values():
        screen = normalize_trace_screen(value)
        trace_store[screen.screen_id] = screen
    persisted = load_persisted_runtime_screens()
    return merge_evidence(trace_store, persisted)

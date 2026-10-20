"""Android platform priors loaded from YAML config.

Provides dialog indicators, repeated-row classes, error patterns,
and package scope rules. All values are Android SDK / AndroidX / Material
Design standard components — NOT app-specific.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = Path(__file__).with_name("android_platform.yaml")


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    """Load and cache the platform priors config."""
    with _DEFAULT_CONFIG.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def get_dialog_indicators() -> dict[str, list[str]]:
    """Get dialog detection indicators (classes + resource_ids)."""
    return _load_config()["dialog_indicators"]


def get_error_patterns() -> list[str]:
    """Get error state name patterns for removal."""
    return _load_config()["error_patterns"]


def get_repeated_row_containers() -> frozenset[str]:
    """Container class leaf names whose direct children are repeated list rows.

    Standard Android/AndroidX scrolling collections only. An identity-less
    clickable inside one of these is a row and may be distinguished by its
    descendants; one outside has no such excuse.
    """
    return frozenset(_load_config()["repeated_row_containers"])


def get_scope_defaults() -> dict[str, Any]:
    """Get scope policy defaults (android_system_packages / system_ui_packages
    / launcher_patterns)."""
    return _load_config()["scope"]

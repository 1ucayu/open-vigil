"""Pydantic configuration models and YAML loader.

Provides typed, validated configuration for all Vigil components.
Loads from YAML files (configs/default.yaml).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Configuration for app exploration."""

    max_exploration_steps: int = 500


class DeviceConfig(BaseModel):
    """Configuration for the target Android device.

    ``type`` is descriptive metadata that is logged and used to tag output
    artifacts (bundle directories, exploration traces). ``serial`` lets
    users pin a specific device when multiple are visible to ADB; when
    ``None``, the resolver in ``vigil.core.device_resolver`` picks one
    deterministically based on ``type``.

    Attributes:
        type: Device kind to target. ``"auto"`` accepts whatever single
            device is visible; ``"emulator"``/``"physical"`` filter the
            ADB device list before selection.
        serial: Explicit ADB serial. When set, bypasses the resolver
            entirely and trusts the value.
    """

    type: Literal["emulator", "physical", "auto"] = "auto"
    serial: str | None = None


class LLMConfig(BaseModel):
    """Connection settings for neural construction and contract enrichment.

    Explicit YAML values override environment defaults. ``VIGIL_LLM_MODEL``
    selects the model; credentials can use ``VIGIL_LLM_API_KEY`` or the standard
    ``OPENAI_API_KEY``. The symbolic execution-time shield never uses this client.

    Attributes:
        provider: API compatibility mode used by the client.
        temperature: Optional sampling temperature.
        base_url: OpenAI-compatible API root, including its version prefix.
        models_url: Optional model-metadata endpoint; defaults to ``base_url/models``.
        api_key: Authentication key; empty for servers that require no authentication.
        model: Model identifier supplied by the configured server.
    """

    provider: Literal["openai_compatible"] = "openai_compatible"
    temperature: float | None = None
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            "VIGIL_LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
    )
    models_url: str | None = Field(default_factory=lambda: os.getenv("VIGIL_LLM_MODELS_URL"))
    api_key: str = Field(
        default_factory=lambda: os.getenv("VIGIL_LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        repr=False,
    )
    model: str = Field(default_factory=lambda: os.getenv("VIGIL_LLM_MODEL", ""))


class VigilConfig(BaseModel):
    """Root configuration model for Vigil.

    Loads from configs/default.yaml.
    """

    app: AppConfig = Field(default_factory=AppConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> VigilConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the config YAML (e.g., configs/default.yaml).

        Returns:
            VigilConfig instance.
        """
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        return cls(**data)

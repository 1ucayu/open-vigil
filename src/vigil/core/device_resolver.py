"""Deterministic ADB device serial resolution.

When multiple devices are visible to ADB (a common state during
development — e.g., a running AVD plus a real phone plugged in for
charging), we don't want Vigil to silently pick whichever device shows
up first. This module resolves a single serial from a ``DeviceConfig``
using explicit precedence: a pinned serial wins, otherwise we filter
candidates by device type and require an unambiguous result.
"""

from __future__ import annotations

import re

from adbutils import adb
from loguru import logger

from vigil.core.config import DeviceConfig

_EMULATOR_SERIAL_RE = re.compile(r"^emulator-\d+$")


def _is_emulator(serial: str) -> bool:
    """Return ``True`` if a serial looks like an emulator serial.

    The convention ``emulator-NNNN`` is set by the Android emulator
    binary; physical devices report their hardware serial which never
    matches this pattern.
    """
    return bool(_EMULATOR_SERIAL_RE.match(serial))


def resolve_device_serial(device_config: DeviceConfig) -> str:
    """Resolve a single ADB device serial from a ``DeviceConfig``.

    Resolution precedence:

    1. If ``device_config.serial`` is set, return it verbatim — no ADB
       lookup, no validation. The user pinned a serial; ADB will fail
       loudly later if it's wrong.
    2. Otherwise query the ADB device list and filter by
       ``device_config.type`` (``"emulator"`` keeps only
       ``emulator-NNNN`` serials; ``"physical"`` keeps only the rest;
       ``"auto"`` keeps everything).
    3. Exactly one surviving candidate → return its serial.
    4. Zero candidates → ``RuntimeError`` describing the filter and the
       devices that were visible.
    5. More than one candidate → ``RuntimeError`` listing them and
       instructing the user to pin ``device.serial`` or pass
       ``--serial``.

    Args:
        device_config: The ``device`` block from a loaded ``VigilConfig``.

    Returns:
        The selected ADB serial string.

    Raises:
        RuntimeError: When the device list is empty for the chosen
            filter, or when more than one candidate matches.
    """
    if device_config.serial is not None:
        logger.info(
            f"Device resolver: using pinned serial={device_config.serial} "
            f"(type={device_config.type})"
        )
        return device_config.serial

    visible = [d.serial for d in adb.device_list()]

    if device_config.type == "emulator":
        candidates = [s for s in visible if _is_emulator(s)]
    elif device_config.type == "physical":
        candidates = [s for s in visible if not _is_emulator(s)]
    else:
        candidates = list(visible)

    if not candidates:
        raise RuntimeError(
            f"No ADB devices match type={device_config.type!r}. "
            f"Visible devices: {visible or '(none)'}. "
            "Connect a device or start an emulator and retry."
        )

    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous device selection: type={device_config.type!r} "
            f"matched {len(candidates)} devices: {candidates}. "
            "Set `device.serial` in your config or pass --serial on the CLI."
        )

    selected = candidates[0]
    logger.info(
        f"Device resolver: type={device_config.type} → selected {selected} "
        f"(1 of {len(visible)} visible)"
    )
    return selected

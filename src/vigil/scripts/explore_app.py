"""CLI entry point: vigil-explore.

Explores an Android app and generates exploration traces.

Usage:
    vigil-explore --app com.example.app --serial DEVICE_SERIAL --steps 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


def main() -> None:
    """Run the app exploration pipeline."""
    parser = argparse.ArgumentParser(
        prog="vigil-explore",
        description="Explore an Android app and generate exploration traces.",
    )
    parser.add_argument(
        "--app",
        required=True,
        help="Android package name (e.g., com.example.app)",
    )
    parser.add_argument(
        "--serial",
        default=None,
        help="ADB device serial (default: require one matching device)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help=(
            "Max observations. Each observation is an independent "
            "cold-start + nav-path replay. (default: 100)"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/apps/<app_name>/)",
    )
    parser.add_argument(
        "--prior-from-device",
        action="store_true",
        help="Extract Activity prior from connected device via 'adb dumpsys'",
    )
    parser.add_argument(
        "--device-type",
        choices=["emulator", "physical", "auto"],
        default=None,
        help="Filter candidate devices by type when --serial is not given",
    )
    from vigil.neuro.explorer_snapshot_dfs import STRATEGY_NAMES

    parser.add_argument(
        "--strategy",
        choices=list(STRATEGY_NAMES),
        default=STRATEGY_NAMES[0],
        help=(
            "Exploration strategy. 'bfs_restart_per_action' (default) restarts "
            "the app and replays a nav path for every action. "
            "'dfs_snapshot_restore' expands a state fully in one visit and "
            "returns to it with an emulator snapshot restore. "
            f"(default: {STRATEGY_NAMES[0]})"
        ),
    )

    args = parser.parse_args()

    # Load config
    from vigil.core.config import VigilConfig

    config_path = Path(args.config)
    if config_path.exists():
        config = VigilConfig.from_yaml(config_path)
    elif args.config != "configs/default.yaml":
        parser.error(f"Configuration file not found: {config_path}")
    else:
        config = VigilConfig()

    # Apply CLI overrides
    config.app.max_exploration_steps = args.steps
    if args.device_type is not None:
        config.device.type = args.device_type

    # Resolve device serial (CLI --serial > config.device.serial > type-filtered auto)
    serial = args.serial
    if serial is None:
        from vigil.core.device_resolver import resolve_device_serial

        try:
            serial = resolve_device_serial(config.device)
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)

    logger.info(f"Target device: serial={serial} type={config.device.type}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else None

    # Extract Activity prior (optional). The prior JSON is persisted to
    # ``static/`` under the app's data dir BEFORE exploration starts so
    # downstream stages can auto-discover it and a crashed exploration
    # still leaves the prior on disk.
    app_prior = None
    if args.prior_from_device:
        from vigil.neuro.app_prior import AppPriorExtractor

        app_name = args.app.replace(".", "_")
        app_dir = output_dir if output_dir is not None else Path(f"data/apps/{app_name}")
        static_dir = app_dir / "static"

        # Androguard path (APK pull + parse)
        app_prior = AppPriorExtractor().extract_from_device_serial(serial, args.app)
        logger.info(
            f"Prior loaded from device: {len(app_prior.activities)} activities, "
            f"{len(app_prior.string_constants)} strings, "
            f"{len(app_prior.widget_declarations)} widgets"
        )

        if app_prior is not None:
            try:
                app_prior.save(static_dir)
                logger.info(f"Cached app prior + resources under {static_dir}")
            except OSError as exc:
                logger.warning(f"Failed to cache app prior: {exc}")

    from vigil.neuro.explorer_snapshot_dfs import build_explorer

    explorer = build_explorer(
        args.strategy,
        device_serial=serial,
        app_package=args.app,
        config=config,
        output_dir=output_dir,
    )

    result = explorer.explore()

    logger.info(
        f"Done: {result.total_steps} steps, {result.unique_screens} screens, "
        f"{result.duration_seconds}s — saved to {result.output_dir}"
    )


if __name__ == "__main__":
    main()

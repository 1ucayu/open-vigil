"""CLI entry point: vigil-build.

Builds an FSM from exploration traces, optionally running the full pipeline:
  Stage 0  → App prior discovery (cached prior or --prior-from-device)
  Stage 1-3 → FSM construction from trace
  Stage 2.5 → Semantic grounding (--ground)
  Stage 4  → typed guard / invariant generation and admission
             (--generate-guards / --generate-invariants)

Usage:
    vigil-build --trace <trace.json>
    vigil-build --trace <trace.json> --ground
    vigil-build --trace <trace.json> --generate-guards
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger


def main() -> None:
    """Run the FSM construction pipeline."""
    parser = argparse.ArgumentParser(
        prog="vigil-build",
        description="Build an FSM from exploration traces.",
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to exploration trace JSON file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: artifacts/<app>/<stage>/fsm.json or efsm.json)",
    )
    parser.add_argument(
        "--app",
        default=None,
        help="Override app package name (auto-detected from trace)",
    )
    parser.add_argument(
        "--include-self-loops",
        action="store_true",
        help="Include self-loop transitions (action doesn't change state)",
    )
    parser.add_argument(
        "--generate-guards",
        action="store_true",
        help="Generate and admit typed transition guards using the LLM (Stage 4)",
    )
    parser.add_argument(
        "--generate-invariants",
        action="store_true",
        help="Generate and admit typed state invariants using the LLM (Stage 4)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip screenshot input for guard generation (text-only prompts)",
    )
    parser.add_argument(
        "--guard-max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum whole-source-state attempts for the LLM guard path (>0). "
            "Defaults to the guard pipeline default."
        ),
    )
    parser.add_argument(
        "--fsm",
        default=None,
        help="Load existing FSM JSON instead of building from trace (use with --generate-guards)",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Configuration YAML; environment defaults apply when the default file is absent",
    )
    parser.add_argument(
        "--visual-ground",
        action="store_true",
        help="Annotate screenshots with visual evidence before guardrail synthesis",
    )
    parser.add_argument(
        "--ground",
        action="store_true",
        help="Run Stage 2.5 semantic grounding after FSM construction",
    )
    parser.add_argument(
        "--prior-from-device",
        default=None,
        metavar="SERIAL",
        help="Extract Activity prior from connected device (provide ADB serial). "
        "For system apps without standalone manifest.",
    )
    parser.add_argument(
        "--apk-dir",
        default=None,
        help="Path to apktool-decompiled APK directory for resource extraction",
    )

    args = parser.parse_args()

    if args.guard_max_attempts is not None and args.guard_max_attempts <= 0:
        parser.error("--guard-max-attempts must be greater than zero")

    trace_path = Path(args.trace)
    if not trace_path.exists():
        logger.error(f"Trace file not found: {trace_path}")
        raise SystemExit(1)

    trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
    app_package = args.app or trace_data.get("app_package", "unknown")
    app_name = app_package.replace(".", "_")

    enrich = args.ground or args.visual_ground or args.generate_guards or args.generate_invariants
    bundle_dir = Path("artifacts") / app_name
    stage = "enriched_efsm" if enrich else "explored_fsm"
    filename = "efsm.json" if enrich else "fsm.json"
    output_path = Path(args.output) if args.output else bundle_dir / stage / filename
    if args.fsm and output_path.resolve() == Path(args.fsm).resolve():
        parser.error("--output must differ from --fsm so the input model remains unchanged")
    if output_path.resolve() == trace_path.resolve():
        parser.error(
            "--output must differ from --trace so the observation evidence remains unchanged"
        )

    llm = None
    if enrich:
        from vigil.core.config import VigilConfig
        from vigil.core.llm_client import LlmClient

        config_path = Path(args.config)
        if not config_path.exists() and args.config != "configs/default.yaml":
            parser.error(f"Configuration file not found: {config_path}")
        config = VigilConfig.from_yaml(config_path) if config_path.exists() else VigilConfig()
        llm = LlmClient(config.llm)

    # Stage 0: App prior extraction (optional)
    prior = None
    if args.prior_from_device:
        from vigil.neuro.app_prior import AppPriorExtractor

        prior = AppPriorExtractor().extract_from_device_serial(args.prior_from_device, app_package)
        logger.info(
            f"Stage 0: extracted prior from device for {prior.package_name} "
            f"({len(prior.activities)} activities)"
        )
    else:
        # Auto-discover the cached prior written by ``vigil-explore`` at
        # the canonical ``<app_dir>/static/app_prior.json`` location.
        from vigil.neuro.app_prior import AppPrior

        static_dir = trace_path.parent.parent / "static"
        if (static_dir / "app_prior.json").exists():
            try:
                prior = AppPrior.load(static_dir)
                logger.info(
                    f"Stage 0: loaded cached app prior from {static_dir} "
                    f"({len(prior.activities)} activities)"
                )
            except (OSError, ValueError) as exc:
                logger.warning(f"Failed to load cached prior at {static_dir}: {exc}")

    # Extract APK resources if apk-dir provided
    if args.apk_dir:
        from vigil.neuro.app_prior import AppPriorExtractor

        apk_dir_path = Path(args.apk_dir)
        if apk_dir_path.is_dir():
            if prior is None:
                manifest = apk_dir_path / "AndroidManifest.xml"
                if manifest.exists():
                    prior = AppPriorExtractor().extract_from_manifest_string(
                        manifest.read_text(encoding="utf-8")
                    )
                else:
                    from vigil.neuro.app_prior import AppPrior

                    prior = AppPrior(package_name=app_package)
            AppPriorExtractor().extract_resources(apk_dir_path, prior)
        else:
            logger.warning(f"APK directory not found: {apk_dir_path}")

    # Stages 1-3: Build or load FSM
    if args.fsm:
        from vigil.models.fsm import AppFSM

        fsm_path = Path(args.fsm)
        if not fsm_path.exists():
            logger.error(f"FSM file not found: {fsm_path}")
            raise SystemExit(1)
        fsm = AppFSM.deserialize(fsm_path)
        logger.info(f"Loaded existing FSM from {fsm_path}: {fsm}")
    else:
        from vigil.neuro.fsm_builder import FsmBuilder

        builder = FsmBuilder(app_package=app_package)
        fsm = builder.build_from_trace(
            trace_path=trace_path,
            include_self_loops=args.include_self_loops,
            app_prior=prior,
        )

        if enrich and args.output is None:
            explored_path = bundle_dir / "explored_fsm" / "fsm.json"
            explored_path.parent.mkdir(parents=True, exist_ok=True)
            fsm.serialize(explored_path)

    raw_screens = trace_data.get("screens", {})
    report_dir = output_path.parent

    if args.visual_ground:
        from vigil.neuro.visual_grounder import (
            ground_fsm_visual_annotations,
            write_visual_grounding_report,
        )

        visual_report = ground_fsm_visual_annotations(fsm, raw_screens, llm, prior)
        write_visual_grounding_report(visual_report, report_dir / "visual_grounding.json")

    # Stage 2.5: Semantic grounding (optional)
    if args.ground:
        from vigil.neuro.semantic_grounder import SemanticGrounder

        grounder = SemanticGrounder(llm)
        fsm = grounder.ground_all_states(fsm, raw_screens, prior)
        logger.info("Stage 2.5: semantic grounding complete")

    # Stage 4: contract-first guard / invariant generation (optional)
    guard_ok = True
    guard_states_failed: list[str] = []
    invariant_states_failed: list[str] = []
    if args.generate_guards or args.generate_invariants:
        if args.generate_guards:
            from vigil.neuro.guard_generation_pipeline import (
                DEFAULT_GUARD_MAX_ATTEMPTS,
                generate_contract_guards,
                write_guard_generation_report,
            )

            guard_outcome = generate_contract_guards(
                fsm,
                raw_screens,
                prior,
                llm=llm,
                guard_use_images=not args.no_images,
                guard_max_attempts=(
                    args.guard_max_attempts
                    if args.guard_max_attempts is not None
                    else DEFAULT_GUARD_MAX_ATTEMPTS
                ),
            )
            guard_ok = guard_outcome.succeeded
            guard_states_failed = [
                outcome.source_state_id
                for outcome in guard_outcome.state_outcomes
                if not outcome.admitted
            ]
            write_guard_generation_report(
                guard_outcome.report,
                report_dir / "transition_guard_generation.json",
            )

        if args.generate_invariants:
            from vigil.neuro.invariant_generation_pipeline import (
                generate_contract_invariants,
                write_invariant_generation_report,
            )

            invariant_report = generate_contract_invariants(
                fsm,
                raw_screens,
                prior,
                invariant_source="llm",
                llm=llm,
                use_images=not args.no_images,
            )
            invariant_states_failed = [
                row["state_id"] for row in invariant_report if row["generation_status"] == "failed"
            ]
            write_invariant_generation_report(
                invariant_report,
                report_dir / "invariant_generation.json",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fsm.serialize(output_path)
    logger.info(f"FSM saved to {output_path}")
    logger.info(f"Summary: {fsm}")
    if not guard_ok:
        logger.warning(f"Guard generation failed for states {guard_states_failed}")
    if invariant_states_failed:
        logger.warning(f"Invariant generation failed for states {invariant_states_failed}")
    if not guard_ok or invariant_states_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

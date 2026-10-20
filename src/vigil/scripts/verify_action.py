"""CLI entry point: vigil-verify-action.

Inspect an EFSM or verify a proposed action without executing it.

Usage:
    # Verify a single action by state ID:
    vigil-verify-action --fsm artifacts/app/enriched_efsm/efsm.json \\
        --state s_001 --action '{"type": "click"}' --goal s_003

    # Verify with a frozen mobile-agent binding snapshot:
    vigil-verify-action --fsm artifacts/app/enriched_efsm/efsm.json \\
        --state s_005 --action '{"type": "click", "target": "e_0042"}' \\
        --bindings '{"wifi_name":"Office"}'

    # Verify through the full runtime pre-action path (binding, clause
    # activation, temporary DSL evaluation, and structural verification):
    vigil-verify-action --fsm artifacts/app/enriched_efsm/efsm.json \\
        --screen screen.xml --action '{"type": "click", "target": "e_0042"}' \\
        --runtime

    # List all states and transitions (inspect mode):
    vigil-verify-action --fsm artifacts/app/enriched_efsm/efsm.json --inspect
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

# ANSI color codes
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_DIM = "\033[2m"


def _resolve_goal(fsm, goal_str: str) -> str:
    """Resolve goal by state name first, fall back to state_id."""
    for state in fsm.states.values():
        if state.name == goal_str:
            return state.state_id
    return goal_str


def _state_label(fsm, state_id: str | None) -> str:
    """Format a state as 'StateName (state_id)'."""
    if state_id is None:
        return "unknown"
    state = fsm.states.get(state_id)
    if state:
        return f"{state.name} ({state_id})"
    return state_id


def _verdict_tag(result) -> str:
    if result.result.value == "allow":
        return f"{_GREEN}✓ ALLOW{_RESET}"
    # A refusal renders in two shades: red for one the model proved (DENY), yellow
    # for one it merely could not rule out (UNCERTAIN).
    if result.proven_refutation:
        return f"{_RED}✗ DENY{_RESET}"
    return f"{_YELLOW}? UNCERTAIN{_RESET}"


def _print_result(fsm, result) -> None:
    """Print a single verification result with ANSI colors."""
    print(f"\n  {_verdict_tag(result)} — {result.reason.value}")
    if result.current_state_id:
        print(f"    Current: {_state_label(fsm, result.current_state_id)}")
    if result.target_state_id:
        print(f"    Target:  {_state_label(fsm, result.target_state_id)}")
    if result.details:
        print(f"    Details: {_DIM}{result.details}{_RESET}")


def _print_runtime_result(fsm, outcome) -> None:
    """Print a RuntimeVerificationResult with ANSI colors."""
    final = outcome.final_result
    print(f"\n  {_verdict_tag(final)} — {final.reason.value}")
    print(f"    Binding: {outcome.binding_status.value}  {_DIM}{outcome.binding_reason}{_RESET}")
    print(f"    Pre-state:  {_state_label(fsm, outcome.pre_state_id)}")
    if outcome.expected_target_state_id:
        print(f"    Expected target: {_state_label(fsm, outcome.expected_target_state_id)}")
    if outcome.observed_target_state_id:
        print(f"    Observed target: {_state_label(fsm, outcome.observed_target_state_id)}")
    if final.details:
        print(f"    Details: {_DIM}{final.details}{_RESET}")


def _cmd_inspect(fsm) -> None:
    """Print all FSM states and transitions."""
    print(f"\n{_BOLD}FSM: {fsm.app_package}{_RESET}")
    print(f"  States: {len(fsm.states)}")
    print(f"  Transitions: {len(fsm.transitions)}")
    if fsm.initial_state:
        print(f"  Initial: {_state_label(fsm, fsm.initial_state)}")

    print(f"\n{_BOLD}States:{_RESET}")
    for state in fsm.states.values():
        activity_name = state.android_context.activity_name
        activity = f" @ {activity_name}" if activity_name else ""
        print(f"  {_CYAN}{state.state_id}{_RESET}  {state.name}{activity}")
        if state.invariant_specs:
            for spec in state.invariant_specs:
                meta = f"  ({spec.source}, observations={spec.evidence_count})"
                print(f"    invariant: {spec.expr}{meta}")

    print(f"\n{_BOLD}Transitions:{_RESET}")
    for transition in fsm.transitions:
        action_type = transition.action.get("type", "?")
        guard_str = f"  guard={transition.guard}" if transition.guard else ""
        print(f"  {transition.source} → {transition.target}  [{action_type}]{guard_str}")

    if fsm.construction_log:
        print(f"\n{_BOLD}Construction Log ({len(fsm.construction_log)} entries):{_RESET}")
        for entry in fsm.construction_log:
            print(f"  {entry.get('action', 'diagnostic')}  {entry.get('reason', '')}")

    if fsm.evolution_log:
        print(f"\n{_BOLD}Evolution Log ({len(fsm.evolution_log)} entries):{_RESET}")
        for entry in fsm.evolution_log:
            print(
                f"  {entry.get('operation', 'unknown')} / "
                f"{entry.get('outcome', 'unknown')}  "
                f"event={entry.get('event', '')}  "
                f"revision={entry.get('efsm_revision_before', '?')}→"
                f"{entry.get('efsm_revision_after', '?')}"
            )


def _parse_verify_inputs(fsm, args):
    """Parse the action/goal/bindings inputs shared by both verify modes."""
    from vigil.models.intent_binding import IntentBindingSnapshot

    action = json.loads(args.action)
    goal = _resolve_goal(fsm, args.goal) if args.goal else None
    binding_snapshot = IntentBindingSnapshot.model_validate(
        json.loads(args.bindings) if args.bindings else {}
    )
    return action, goal, binding_snapshot


def _cmd_runtime_verify(fsm, args) -> None:
    """Verify a single action through the full runtime-monitor pre-action path."""
    from vigil.runtime.monitor import RuntimeMonitor
    from vigil.runtime.screen_capture import raw_screen_from_xml
    from vigil.runtime.session import RuntimeSession

    screen_path = Path(args.screen)
    if not screen_path.exists():
        print(f"{_RED}Error: screen XML not found: {screen_path}{_RESET}", file=sys.stderr)
        sys.exit(1)

    action, goal, binding_snapshot = _parse_verify_inputs(fsm, args)
    screen = raw_screen_from_xml(
        screen_path.read_text(encoding="utf-8"),
        screen_id=screen_path.stem,
        app_package=fsm.app_package,
        activity_name=args.activity,
        package_name=args.package,
    )

    session = RuntimeSession(binding_snapshot=binding_snapshot)
    monitor = RuntimeMonitor(fsm)
    outcome = monitor.verify_action(screen, action, session=session, goal_state=goal)

    _print_runtime_result(fsm, outcome)


def _cmd_verify(fsm, args) -> None:
    """Verify a single action."""
    from vigil.symbolic.decision_engine import DecisionEngine

    action, goal, binding_snapshot = _parse_verify_inputs(fsm, args)
    engine = DecisionEngine(fsm)

    if args.state:
        result = engine.verify_by_state(
            args.state,
            action,
            goal_state=goal,
            binding_snapshot=binding_snapshot,
        )
    elif args.screen:
        from vigil.runtime.screen_capture import raw_screen_from_xml

        xml_path = Path(args.screen)
        if not xml_path.exists():
            print(f"{_RED}Error: screen XML not found: {xml_path}{_RESET}", file=sys.stderr)
            sys.exit(1)
        xml_text = xml_path.read_text(encoding="utf-8")
        screen = raw_screen_from_xml(
            xml_text,
            screen_id=xml_path.stem,
            app_package=fsm.app_package,
            activity_name=args.activity,
            package_name=args.package,
        )
        result = engine.verify(
            screen,
            action,
            goal_state=goal,
            binding_snapshot=binding_snapshot,
        )
    else:
        print(f"{_RED}Error: --state or --screen required{_RESET}", file=sys.stderr)
        sys.exit(1)

    _print_result(fsm, result)


def main() -> None:
    """Vigil runtime verification CLI."""
    parser = argparse.ArgumentParser(
        prog="vigil-verify-action",
        description="Verify a proposed action against a Vigil FSM.",
    )
    parser.add_argument("--fsm", required=True, help="Path to FSM JSON")

    # Input mode
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--state", help="Current FSM state ID")
    input_group.add_argument("--screen", help="Path to screen XML for auto-localization")
    input_group.add_argument("--inspect", action="store_true", help="List all states/transitions")
    parser.add_argument("--activity", help="Captured foreground activity for --screen")
    parser.add_argument("--package", help="Captured foreground package for --screen")

    # Action specification
    parser.add_argument("--action", help='Action JSON (e.g., \'{"type": "click"}\')')
    # Optional context
    parser.add_argument("--goal", help="Goal state ID or name")
    parser.add_argument(
        "--bindings",
        help=('Frozen binding snapshot JSON (e.g., \'{"wifi_name": "Office"}\')'),
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Verify via the full runtime-monitor pre-action path (requires --screen).",
    )
    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    # Load FSM
    from vigil.models.fsm import AppFSM

    fsm_path = Path(args.fsm)
    if not fsm_path.exists():
        print(f"{_RED}Error: FSM file not found: {fsm_path}{_RESET}", file=sys.stderr)
        sys.exit(1)
    fsm = AppFSM.deserialize(fsm_path)

    # Dispatch
    if args.inspect:
        _cmd_inspect(fsm)
    elif args.runtime:
        if not args.screen or not args.action:
            msg = f"{_RED}Error: --runtime requires --screen and --action{_RESET}"
            print(msg, file=sys.stderr)
            sys.exit(1)
        _cmd_runtime_verify(fsm, args)
    elif args.action:
        _cmd_verify(fsm, args)
    else:
        msg = f"{_RED}Error: --action or --inspect required{_RESET}"
        print(msg, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

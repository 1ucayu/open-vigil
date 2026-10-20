"""Depth-first exploration with optional emulator checkpoint restoration.

This strategy inherits device interaction, action enumeration, observation capture,
and trace serialization from :class:`vigil.neuro.explorer.AppExplorer`. It expands
all untried actions of a state before moving to another state and uses a single
emulator checkpoint to restore the source between sibling actions. Newly discovered
states are visited in depth-first order.

Only one checkpoint is retained at a time. When snapshots are unavailable, disk
space is insufficient, or restoration fails, navigation uses the inherited app
restart and recorded action path. Actions that belong to transient UI sub-states
are left for enumeration in those states if they cannot resolve from the checkpoint.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any

import uiautomator2 as u2
from loguru import logger

from vigil.models.state import (
    SENTINEL_ACTION_FAILED,
    SENTINEL_COLD_START_FAILED,
    SENTINEL_LEFT_APP,
    SENTINEL_TARGETS,
    RawScreen,
)
from vigil.neuro.explorer import (
    AppExplorer,
    ExplorationResult,
    ExplorationTrace,
    action_key,
)

#: Console name of the single live checkpoint. Reused for every state, so the
#: emulator never holds more than one Vigil snapshot.
CHECKPOINT_NAME = "vigil_dfs_ckpt"

#: Minimum free disk space required before saving an emulator checkpoint.
#: Below this threshold, navigation uses the inherited restart and replay path.
MIN_FREE_BYTES = 8 * 1024**3


class SnapshotDfsExplorer(AppExplorer):
    """Depth-first exploration that returns to a state by snapshot restore.

    Inherits every device primitive and the whole serializer from
    :class:`AppExplorer`; only :meth:`explore` is replaced. The emitted trace is
    the same schema, with ``exploration_strategy`` restamped.
    """

    STRATEGY_NAME: str = "dfs_snapshot_restore"

    #: Maximum duration of one emulator snapshot console call.
    SNAPSHOT_TIMEOUT: float = 180.0

    #: Minimum untried sibling actions needed to amortize checkpoint creation.
    #: States below this threshold use the inherited restart and replay path.
    CHECKPOINT_MIN_ACTIONS: int = 3

    #: How long to wait for the guest's uiautomator2 agent after a restore.
    RECONNECT_TIMEOUT: float = 30.0
    RECONNECT_INTERVAL: float = 0.5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # state_id the live checkpoint holds, or None when there is none.
        self._checkpoint_state: str | None = None
        self._snapshot_counts: dict[str, int] = {
            "snapshot_saves": 0,
            "snapshot_restores": 0,
            "snapshot_failures": 0,
            "snapshot_skipped_disk": 0,
            "replay_navigations": 0,
        }

    # ------------------------------------------------------------- snapshots

    def _emu_console(self, *verb: str) -> bool:
        """Run one ``adb -s <serial> emu ...`` console verb.

        The console is reached through adb rather than uiautomator2 because it
        addresses the *emulator*, not the guest. ``adb`` is resolved from PATH;
        when it is absent the strategy degrades rather than failing the run.
        """
        adb = shutil.which("adb")
        if adb is None:
            logger.warning("adb not on PATH; snapshot restore unavailable")
            return False
        try:
            proc = subprocess.run(
                [adb, "-s", self._serial, "emu", *verb],
                capture_output=True,
                text=True,
                timeout=self.SNAPSHOT_TIMEOUT,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning(f"snapshot console {' '.join(verb)} failed: {exc}")
            return False
        # The console prints "OK" on success; a failure prints a diagnostic and
        # still exits 0, so the return code alone is not a verdict.
        return proc.returncode == 0 and "OK" in proc.stdout

    def _disk_headroom_ok(self) -> bool:
        try:
            return shutil.disk_usage("/").free >= MIN_FREE_BYTES
        except OSError:
            return False

    def _save_checkpoint(self, state_id: str) -> bool:
        """Pin the device's current configuration as the checkpoint for ``state_id``."""
        if not self._disk_headroom_ok():
            self._snapshot_counts["snapshot_skipped_disk"] += 1
            logger.warning("free disk below floor; skipping checkpoint")
            return False
        t0 = time.monotonic()
        if not self._emu_console("avd", "snapshot", "save", CHECKPOINT_NAME):
            self._snapshot_counts["snapshot_failures"] += 1
            self._checkpoint_state = None
            return False
        self._checkpoint_state = state_id
        self._snapshot_counts["snapshot_saves"] += 1
        logger.debug(f"checkpoint saved for {state_id[:6]} in {time.monotonic() - t0:.2f}s")
        return True

    def _restore_checkpoint(self, state_id: str) -> bool:
        """Return the device to ``state_id``'s checkpoint.

        On success the caller may treat the device as being at ``state_id``
        exactly, including app data. On failure the checkpoint is dropped so the
        caller falls back to the inherited cold-start + replay path.
        """
        if self._checkpoint_state != state_id:
            return False
        t0 = time.monotonic()
        if not self._emu_console("avd", "snapshot", "load", CHECKPOINT_NAME):
            self._snapshot_counts["snapshot_failures"] += 1
            self._checkpoint_state = None
            return False
        self._snapshot_counts["snapshot_restores"] += 1
        # A restore rewinds the *guest*, including the uiautomator2 agent that
        # the host holds an open HTTP connection to. That connection does not
        # survive: the first call after a restore raises RemoteDisconnected and
        # every subsequent action resolves against a dead session, which
        # surfaces as a run of ACTION_FAILED rows rather than as an error.
        # Re-establishing the session is therefore part of restoring, not an
        # optimisation.
        if not self._reconnect_after_restore():
            self._checkpoint_state = None
            return False
        # The restored device is bit-identical to the save, so the viewport is
        # exactly where it was and the warm-enumeration gate is sound again.
        self._device_scrolled = False
        self._current_state_id = state_id
        logger.debug(f"checkpoint restored for {state_id[:6]} in {time.monotonic() - t0:.2f}s")
        return True

    def _reconnect_after_restore(self) -> bool:
        """Re-establish the uiautomator2 session, and wait for it to answer.

        The agent inside the guest needs a moment to accept connections again
        after a rewind, so a single reconnect is not enough; poll until
        ``info`` succeeds. Returns False when the agent never comes back, which
        drops the checkpoint and sends the caller down the cold-start path.
        """
        deadline = time.monotonic() + self.RECONNECT_TIMEOUT
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._device = u2.connect(self._serial)
                self._device.info  # noqa: B018 - the probe IS the assertion
            except Exception as exc:  # noqa: BLE001 - any transport error retries
                last_exc = exc
                time.sleep(self.RECONNECT_INTERVAL)
                continue
            return True
        logger.warning(f"uiautomator2 did not return after restore: {last_exc}")
        self._snapshot_counts["snapshot_failures"] += 1
        return False

    def _drop_checkpoint(self) -> None:
        if self._checkpoint_state is None:
            return
        self._emu_console("avd", "snapshot", "delete", CHECKPOINT_NAME)
        self._checkpoint_state = None

    # ------------------------------------------------------------------ loop

    def explore(self) -> ExplorationResult:  # noqa: C901 - mirrors the base loop's shape
        budget = self._config.app.max_exploration_steps
        logger.info(f"Budget: {budget} observations — strategy={self.STRATEGY_NAME}")
        started = time.monotonic()
        try:
            self._connect_device()
            if not self._cold_start_app():
                raise RuntimeError(f"Could not launch {self._app_package}")

            entry = self._capture_screen()
            if entry is None:
                raise RuntimeError("Initial capture failed after cold-start")
            entry_sid = entry.get_hybrid_state_id(self._app_package)
            self._current_state_id = entry_sid
            self._nav_paths[entry_sid] = []
            self._verified_navigation_states.add(entry_sid)

            screens: dict[str, RawScreen] = {entry.screen_id: entry}
            traces: list[ExplorationTrace] = []
            counts = {"cold_start_failures": 0, "left_app": 0, "action_failures": 0}
            done = 0

            # Depth-first frontier. LIFO, so the traversal descends before it
            # broadens; ``_nav_paths`` still records a replay path for every
            # state, because moving between states uses the inherited path.
            stack: list[str] = [entry_sid]
            opened: set[str] = set()

            while done < budget and stack:
                state_id = stack.pop()
                if state_id in opened:
                    continue
                opened.add(state_id)

                # --- get to the state, and enumerate it there ---------------
                if not self._ensure_at(state_id):
                    logger.info(f"state={state_id[:6]} unreachable; skipping")
                    continue
                if state_id not in self._all_actions_per_state:
                    actions = self._enumerate_actions_for_state(state_id, self._nav_paths[state_id])
                    self._all_actions_per_state[state_id] = {action_key(a): a for a in actions}
                    logger.info(f"state={state_id[:6]} enumerated {len(actions)} actions")

                # Enumeration sweeps and scrolls, so the device is no longer
                # necessarily at the enumerated screen. Re-establish, then pin.
                if not self._ensure_at(state_id):
                    continue
                if self._unexplored_count(state_id) >= self.CHECKPOINT_MIN_ACTIONS:
                    self._save_checkpoint(state_id)

                # --- expand every action of this state, in one visit --------
                discovered: list[str] = []
                while done < budget:
                    action = self._pick_best_in_state(state_id)
                    if action is None:
                        break
                    key = action_key(action)

                    # Every sibling after the first starts from the checkpoint.
                    # A failed restore is not fatal: _perform_one_observation
                    # falls back to cold-start + replay on its own, exactly as
                    # the base strategy always does.
                    from_checkpoint = False
                    if self._current_state_id != state_id:
                        from_checkpoint = self._restore_checkpoint(state_id)
                        if not from_checkpoint:
                            self._snapshot_counts["replay_navigations"] += 1

                    done += 1
                    t0 = time.monotonic()
                    trace, pre_screen, post_screen = self._perform_one_observation(
                        intended_source_state_id=state_id,
                        intended_nav_path=self._nav_paths[state_id],
                        action=action,
                        step=done,
                    )
                    self._log_step(trace, len(self._nav_paths[state_id]), time.monotonic() - t0)
                    traces.append(trace)

                    requeued = self._retire_or_requeue(state_id, key, trace)
                    if requeued:
                        counts["unexecuted_requeued"] = counts.get("unexecuted_requeued", 0) + 1
                    if from_checkpoint and trace.target_state_id == SENTINEL_ACTION_FAILED:
                        counts["unresolvable_from_checkpoint"] = (
                            counts.get("unresolvable_from_checkpoint", 0) + 1
                        )
                    self._account_sentinels(state_id, key, trace, counts, requeued)

                    if pre_screen is not None:
                        screens[pre_screen.screen_id] = pre_screen
                    if post_screen is not None:
                        screens[post_screen.screen_id] = post_screen

                    new_state = self._register_target(state_id, trace)
                    if new_state is not None:
                        discovered.append(new_state)

                # --- close the state, descend into what it revealed ---------
                self._drop_checkpoint()
                # Reversed so the first-discovered target is explored first once
                # the stack pops; the traversal is still depth-first.
                for sid in reversed(discovered):
                    if sid not in opened:
                        stack.append(sid)

            if not stack and done < budget:
                logger.info("Exploration complete: no more unexplored actions")

            self._log_summary()
            duration = time.monotonic() - started
            counts.update(self._snapshot_counts)
            return self._finalize(traces, screens, self._nav_paths, counts, done, duration)
        finally:
            self._drop_checkpoint()
            self._restore_animation_scales()

    # -------------------------------------------------------------- helpers

    def _unexplored_count(self, state_id: str) -> int:
        """How many enumerated actions at ``state_id`` are still worth trying."""
        candidates = self._all_actions_per_state.get(state_id, {})
        spent = self._explored_per_state[state_id] | self._blacklisted_per_state[state_id]
        return sum(1 for key in candidates if key not in spent)

    def _ensure_at(self, state_id: str) -> bool:
        """Put the device at ``state_id``, preferring a restore over a replay.

        Returns False only when the inherited cold start itself fails; a replay
        that lands elsewhere is caught downstream by ``_perform_one_observation``
        exactly as in the base strategy.
        """
        if self._current_state_id == state_id:
            return True
        if self._restore_checkpoint(state_id):
            return True
        self._snapshot_counts["replay_navigations"] += 1
        if not self._cold_start_app():
            self._current_state_id = None
            return False
        for nav_step in self._nav_paths.get(state_id, []):
            self._execute_action(nav_step)
            self._wait_after_action(nav_step, cap=self.SETTLE_CAP)
        self._current_state_id = state_id
        return True

    def _account_sentinels(
        self,
        state_id: str,
        key: str,
        trace: ExplorationTrace,
        counts: dict[str, int],
        requeued: bool,
    ) -> None:
        """Sentinel bookkeeping, matching the base loop's semantics exactly."""
        target = trace.target_state_id
        if target == SENTINEL_LEFT_APP:
            counts["left_app"] += 1
            # Blacklist only what was actually retired: a re-queued row proved
            # the replay never arrived, not that the action is forbidden here.
            if not requeued:
                self._blacklisted_per_state[state_id].add(key)
        elif target == SENTINEL_ACTION_FAILED:
            counts["action_failures"] += 1
        elif target == SENTINEL_COLD_START_FAILED:
            counts["cold_start_failures"] += 1
        if target in SENTINEL_TARGETS:
            # A sentinel row leaves the device somewhere unknown; force the next
            # sibling through a restore rather than trusting the tracker.
            self._current_state_id = None

    def _register_target(self, intended_source: str, trace: ExplorationTrace) -> str | None:
        """Record a successful observation's target; return it when it is new.

        Mirrors the base loop's navigation-path bookkeeping: a drifted source or an
        ANDROID_SYSTEM screen never becomes a navigable state.
        """
        target = trace.target_state_id
        if not target or target in SENTINEL_TARGETS:
            return None
        actual_src = trace.source_state_id
        drifted = bool(actual_src) and actual_src != intended_source
        framework_target = bool(trace.metadata.get("framework_scope"))
        framework_source = trace.metadata.get("scope_pre") == "android_system"
        if drifted:
            self._drift_count += 1
        if actual_src and not drifted and not framework_source:
            self._verified_navigation_states.add(actual_src)

        # The device now sits at the target, so the tracker follows it.
        self._current_state_id = target

        if target in self._nav_paths:
            return None
        if framework_target:
            self._unverified_navigation_targets.add(target)
            return None
        if drifted or framework_source:
            self._unverified_navigation_targets.add(target)
            return None
        self._nav_paths[target] = [*self._nav_paths[intended_source], trace.action]
        self._verified_navigation_states.add(target)
        return target


def build_explorer(strategy: str, **kwargs: Any) -> AppExplorer:
    """Construct the explorer for ``strategy``.

    The default strategy uses breadth-first enumeration with recorded navigation
    paths. The snapshot strategy adds depth-first ordering and checkpoint restoration.
    """
    if strategy == SnapshotDfsExplorer.STRATEGY_NAME:
        return SnapshotDfsExplorer(**kwargs)
    if strategy == AppExplorer.STRATEGY_NAME:
        return AppExplorer(**kwargs)
    raise ValueError(f"unknown exploration strategy: {strategy!r}")


#: Selectable strategy names, in CLI order. The first is the default.
STRATEGY_NAMES: tuple[str, ...] = (AppExplorer.STRATEGY_NAME, SnapshotDfsExplorer.STRATEGY_NAME)


__all__ = ["STRATEGY_NAMES", "SnapshotDfsExplorer", "build_explorer"]

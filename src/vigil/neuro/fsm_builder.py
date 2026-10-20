"""Stage 3: Hierarchical FSM Construction.

Builds an AppFSM from exploration traces and abstract states. Organizes states
into a hierarchy (App > Activity > Fragment > Component) using Android Activity
names from the accessibility tree. Built on networkx.DiGraph.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Hashable
from pathlib import Path
from typing import Any

from loguru import logger

from vigil.core.ui_parser import parse_hierarchy_xml
from vigil.models.action import Action
from vigil.models.fsm import (
    AbstractState,
    AndroidStateContext,
    AppFSM,
    ContainerType,
    HierarchyLevel,
    ProvenanceEntry,
    StateAbstraction,
    StateEvidence,
    StateIdentity,
    SubFsmTemplate,
    Transition,
    canonical_action_key,
    canonical_transition_action,
    transition_identity_key,
)
from vigil.models.state import SENTINEL_TARGETS
from vigil.neuro.app_prior import AppPrior
from vigil.neuro.behavioral_signature import (
    compute_quotient_label,
    signature_hash,
)
from vigil.neuro.state_split import (
    secondary_feature_group_hash,
    secondary_feature_signature,
    with_secondary_feature_hash,
)

# Bounded passes for the post-quotient determinism guard; the block count
# strictly grows on every productive pass, so the fixed point is reached
# well below the bound.
_MAX_DETERMINISM_PASSES = 8

# DFS depth bound for Sub-FSM template subgraph collection — keeps a
# navigate-back edge from swallowing the whole FSM into one template.
_TEMPLATE_SUBGRAPH_MAX_DEPTH = 5

# Identity fields strong enough to make a click self-loop a meaningful,
# verifiable affordance (Settings list rows, switches/checkboxes/radios,
# buttons, parameterised list rows). Used in both _build_transitions and
# _merge_scroll_duplicates so identity-bearing self-loops survive end-to-end.
# Note: ``target`` (the explorer's element_id handle) is intentionally
# excluded — element ids are volatile per-capture and do not bind a stable
# affordance across screens. Stable identity comes from text / resource_id /
# content-desc. Class-only or selector-only identity is too weak for preserved
# click self-loops.
_CLICK_SELF_LOOP_IDENTITY_FIELDS = (
    "target_text",
    "resource_id",
    "target_resource_id",
    "target_content_desc",
)


def _click_action_has_identity(action: dict[str, Any]) -> bool:
    """A click action carries enough identity to act as a verifiable self-loop
    when at least one stable identity field is populated."""
    for field in _CLICK_SELF_LOOP_IDENTITY_FIELDS:
        value = action.get(field)
        if value:
            return True
    return False


def build_trace_action_sample(
    trace: dict[str, Any],
    raw_screens: dict[str, Any] | None = None,
    *,
    source_state: str | None = None,
    target_state: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Build the trace-level action sample used as transition provenance.

    This is the single conversion path from an exploration trace action plus its
    source-screen element evidence into the raw action sample that is later
    canonicalized into the verifier-level transition label.
    """
    action_data = trace.get("action", {}) or {}
    action = Action(**action_data)
    fsm_action = action.to_fsm_dict()
    source_sid = trace.get("source_screen_id", "")
    action_type = (action_data.get("action_type") or action_data.get("type") or "").lower()

    identity_inconsistent = False
    if raw_screens and action.target_element_id:
        source_screen = raw_screens.get(source_sid, {})
        elements = source_screen.get("interactable_elements", source_screen.get("elements", []))
        claimed_rid = (
            (fsm_action.get("target_selector") or {}).get("resource_id")
            or fsm_action.get("target_resource_id")
            or fsm_action.get("resource_id")
            or ""
        )
        resolved_el = None
        if claimed_rid:
            for el in elements:
                if (el.get("resource_id") or "") == claimed_rid:
                    resolved_el = el
                    break
        if resolved_el is None:
            for el in elements:
                if el.get("element_id") == action.target_element_id:
                    el_rid = el.get("resource_id") or ""
                    if claimed_rid and el_rid and el_rid != claimed_rid:
                        identity_inconsistent = True
                        logger.warning(
                            "stale target_element_id at step {}: "
                            "id={} in screen={} resolves to rid={} but "
                            "trace selector/action rid={}; skipping "
                            "identity enrichment",
                            trace.get("step_number"),
                            action.target_element_id,
                            source_sid,
                            el_rid,
                            claimed_rid,
                        )
                        resolved_el = None
                    else:
                        resolved_el = el
                    break
        if resolved_el is not None:
            el = resolved_el
            text = el.get("text") or ""
            desc = el.get("content_description") or ""
            rid = el.get("resource_id") or ""
            cls = el.get("class_name") or ""
            # ``target_text`` records exactly what the trace observed: the
            # element's own text, and nothing when it has none. Falling back to
            # the content-description here would fabricate identity the runtime
            # binder never reproduces (it reads text only), permanently
            # unmatching every icon-only control's edge -- see
            # ``ActionBinder._bind_to_element``.
            candidates = {
                "target_text": text,
                "target_content_desc": desc,
                "target_resource_id": rid,
                "resource_id": rid,
                "target_class": cls,
                "target_class_name": cls,
            }
            for field, candidate in candidates.items():
                if candidate and not (fsm_action.get(field) or ""):
                    fsm_action[field] = candidate
            el_selector = el.get("target_selector") or el.get("selector")
            if el_selector and not fsm_action.get("target_selector"):
                fsm_action["target_selector"] = el_selector

    sel_rid = (fsm_action.get("target_selector") or {}).get("resource_id") or ""
    rid_v = fsm_action.get("resource_id") or ""
    trid_v = fsm_action.get("target_resource_id") or ""
    trace_rid = action_data.get("target_resource_id") or action_data.get("resource_id") or ""
    chosen_rid: str | None = None
    if sel_rid:
        if (rid_v and rid_v != sel_rid) or (trid_v and trid_v != sel_rid):
            identity_inconsistent = True
        chosen_rid = sel_rid
    elif rid_v and trid_v and rid_v != trid_v:
        identity_inconsistent = True
        if trace_rid and trace_rid == rid_v:
            chosen_rid = rid_v
        elif trace_rid and trace_rid == trid_v:
            chosen_rid = trid_v
        else:
            chosen_rid = trid_v
    elif rid_v:
        chosen_rid = rid_v
    elif trid_v:
        chosen_rid = trid_v

    if chosen_rid:
        fsm_action["resource_id"] = chosen_rid
        fsm_action["target_resource_id"] = chosen_rid
    else:
        fsm_action.pop("resource_id", None)
        fsm_action.pop("target_resource_id", None)

    if identity_inconsistent:
        logger.warning(
            "identity_inconsistent transition at step {}: "
            "source={} target={} rid={} trid={} sel_rid={} -> kept={}",
            trace.get("step_number"),
            source_state,
            target_state,
            rid_v,
            trid_v,
            sel_rid,
            chosen_rid,
        )

    if (
        action_type == "click"
        and source_state is not None
        and source_state == target_state
        and _click_action_has_identity(fsm_action)
    ):
        # element_id is capture-local; stable text/resource binds across captures.
        fsm_action.pop("target", None)

    return fsm_action, identity_inconsistent


class FsmBuilder:
    """Build an AppFSM from an exploration trace JSON file.

    Args:
        app_package: Android package name.
    """

    def __init__(self, app_package: str) -> None:
        self._app_package = app_package
        self._raw_screens: dict[str, Any] = {}
        self._refinement_log: list[dict[str, Any]] = []
        self._collapse_redirects: dict[str, str] = {}

    @staticmethod
    def _classify_trace_evidence(metadata: dict[str, Any]) -> tuple[bool, bool]:
        """Return ``(skip, ambiguous_evidence)`` for a trace metadata dict.

        Neither value is a trust modifier -- membership in the transition graph
        represents accepted evidence. ``skip`` controls evidence admission: an
        observation whose endpoints leave the app under test says nothing about
        the app's behaviour and never enters the graph. ``ambiguous_evidence``
        marks an admitted observation whose scope only partly covers the app,
        so offline refinement must not draw a partition conclusion from it.
        Runtime never reads it.
        """
        if not metadata:
            return False, False

        scope_pre = metadata.get("scope_pre")
        scope_post = metadata.get("scope_post")
        scope_pre_value = str(scope_pre) if scope_pre is not None else None
        scope_post_value = str(scope_post) if scope_post is not None else None

        if scope_pre_value == "android_system" or scope_post_value == "android_system":
            return True, False

        # Older trace files called framework scope "low_trust_scope". This is
        # observation-scope metadata, never an executable transition trust gate.
        if metadata.get("framework_scope", metadata.get("low_trust_scope")) is True:
            if scope_pre_value == "in_app" and scope_post_value == "in_app":
                return False, True
            return True, False

        return False, False

    @staticmethod
    def _merge_transition_evidence(existing: Transition, incoming: Transition) -> None:
        """Combine duplicate-edge evidence.

        Attribution rule (a construction rule, not a trust rule): **a single
        cleanly-attributed observation of an edge clears that edge's
        ambiguous-evidence marking, regardless of how many ambiguous
        observations preceded it.**

        Duplicates only reach this function when they agree on the full
        identity ``(source, target, canonical_action_key)``, so a cleanly attributed
        duplicate is direct positive evidence that the edge exists and resolves
        the way the ambiguous sighting said it did. Making the marking sticky
        would permanently exclude from refinement an edge for which the builder
        holds clean, attributable evidence.

        Known limitation — ``ambiguous_evidence`` is one boolean with several
        producers, and this rule can only justify the first:

        1. partial observation *scope* (:meth:`_classify_trace_evidence`) — a
           property of the observation channel, which a clean sighting of the
           same edge genuinely supersedes;
        2. ``identity_inconsistent`` action-sample resolution;
        3. :meth:`_downgrade_conflict`, which records a *structural* doubt
           (indistinguishable source screens reaching different targets).

        For (3) especially, a clean duplicate does not refute the ambiguity, so
        clearing is not obviously right. Separating those producers means giving
        each its own provenance-carried reason rather than overloading one flag;
        that requires distinct provenance fields rather than one shared flag.
        """
        existing.observed_count += incoming.observed_count
        # Provenance lists are append-only; preserve every supporting record so the
        # validator can reconstruct evidence.
        if incoming.provenance:
            existing.provenance.extend(incoming.provenance)
        if existing.ambiguous_evidence != incoming.ambiguous_evidence:
            existing.ambiguous_evidence = False

    def build_from_trace(
        self,
        trace_path: Path,
        include_self_loops: bool = False,
        app_prior: AppPrior | None = None,
    ) -> AppFSM:
        """Build an FSM from a serialized exploration trace.

        Args:
            trace_path: Path to the exploration JSON file.
            include_self_loops: Whether to include transitions where source == target.

        Returns:
            A fully constructed AppFSM.
        """
        self._raw_screens = {}
        self._refinement_log.clear()
        self._collapse_redirects.clear()
        data = json.loads(trace_path.read_text(encoding="utf-8"))
        raw_screens = data.get("screens", {})
        # Drop sentinel traces emitted by the explorer when cold-start or action
        # execution failed, or when the app was left — their target_state_id is a
        # reserved uppercase label, not a real state hash.
        raw_traces = [
            t for t in data.get("traces", []) if t.get("target_state_id") not in SENTINEL_TARGETS
        ]
        # Make raw_screens available to downstream helpers (e.g.,
        # _classify_containers_structural) without threading it as a parameter.
        self._raw_screens = raw_screens

        # Step 1: Build canonical states. When a screen ships ``state_id``
        # (the explorer format), trust it directly — it's a functional,
        # text-anchored identity already. Screens without one dedup on the
        # scroll-aware functional fingerprint instead.
        sid_to_state_id, states = self._build_states(raw_screens, trace_path.parent, app_prior)

        # Step 2: Build transitions from traces
        transitions = self._build_transitions(
            raw_traces, sid_to_state_id, include_self_loops, raw_screens
        )

        # Step 3: Merge duplicate transitions
        transitions = self._merge_transitions(transitions)

        # Step 3.5: APE-style refinement — split or downgrade conflicting successors
        transitions = self._refine_conflicting_successors(
            states, sid_to_state_id, transitions, raw_screens
        )

        # Step 4: Detect initial state (prefers manifest launcher activity)
        initial_state = self._detect_initial_state(
            raw_traces, sid_to_state_id, states=states, app_prior=app_prior
        )

        # Step 5: Disambiguate duplicate state names
        self._disambiguate_names(states)

        # Step 6: Infer hierarchy from activity names
        self._infer_hierarchy(states)

        # Step 7: Assemble FSM
        fsm = AppFSM(app_package=self._app_package)
        fsm.initial_state = initial_state

        for state in states.values():
            fsm.add_state(state)
        for t in transitions:
            fsm.add_transition(t)

        # Flush refinement diagnostics buffered on the builder into the FSM.
        fsm.construction_log.extend(self._refinement_log)
        flushed_refinement_log_count = len(self._refinement_log)

        # Step 8: Post-processing — merge duplicates and remove error states
        merged = self._merge_scroll_duplicates(fsm)
        removed = self._remove_error_states(fsm)
        if merged or removed:
            logger.info(
                f"Post-processing: merged {merged} duplicate states, removed {removed} error states"
            )

        # Step 8.5: Behavioral quotient via trace-observation-compatible
        # partition refinement. States are grouped by their schema-only
        # quotient label (activity, window, dialog flag, action surface
        # with per-instance rids stripped, form coarse status, anchored
        # landmark/label *schema* — no literal title/contact text), then
        # refined while their observed (canonical_action_key ->
        # target_block) maps agree on shared keys. A post-quotient
        # determinism guard splits any block that would still violate
        # the verifier invariant after lifting.
        behavioral_merged = self._coarsen_behavioral_duplicates(fsm, raw_screens)
        if behavioral_merged:
            logger.info(
                f"Behavioral quotient: merged {behavioral_merged} states "
                "(trace-observation-compatible partition refinement)"
            )

        # Step 9: Build Sub-FSM templates for dynamic containers
        structurally_labeled = self._classify_containers_structural(fsm)
        if structurally_labeled:
            logger.info(
                f"Structural container classification: {structurally_labeled} DYNAMIC states"
            )
        templates_created = self._build_sub_fsm_templates(fsm)
        if templates_created:
            logger.info(f"Created {templates_created} Sub-FSM templates")
        fsm.construction_log.extend(self._refinement_log[flushed_refinement_log_count:])

        # Step 10: Detect dialog states and assign hierarchy
        dialogs = self._detect_dialog_states(fsm, raw_screens)
        if dialogs:
            logger.info(f"Detected {dialogs} dialog states (COMPONENT level)")

        # Step 11: record every live-screen identity each state covers, so runtime
        # localization can exact-match a member screen and not only the representative
        # (schema v6). Deliberately last: merges, splits and redirects have all settled,
        # so the member sets describe the partition that is actually written to disk.
        indexed_identities = assign_member_fingerprints(fsm, raw_screens)

        logger.info(
            f"FSM built: {len(fsm.states)} states, {len(fsm.transitions)} transitions, "
            f"{indexed_identities} member identities, initial_state={initial_state}"
        )
        return fsm

    # --- Post-processing: duplicate/error state cleanup ---

    def _merge_scroll_duplicates(self, fsm: AppFSM) -> int:
        """Merge states that share (activity_name, base_name) AND are
        structurally compatible.

        Earlier behavior merged every same-name pair, which silently
        collapsed semantically distinct screens that happened to share a
        toolbar title. The hardened policy requires:

          1. Same ``activity_name``.
          2. Same ``base_name`` (after stripping any ``#N`` scroll suffix).
          3. Compatible structural skeletons: either identical structural
             fingerprints, or fingerprints that agree on the non-scrollable
             skeleton (so the only differences live inside scrollable
             subtrees, which is the genuine scroll-duplicate signal).

        Same-name candidates with incompatible skeletons are kept separate.
        Self-loops produced by redirecting transitions are dropped only
        when they are CLICK no-ops; SCROLL_UP / SCROLL_DOWN / INPUT_TEXT
        and toggle self-loops are preserved because they remain legal
        affordances after the merge.

        Returns:
            Number of states merged away.
        """
        import re

        groups: dict[tuple[str | None, str], list[str]] = defaultdict(list)
        for state in fsm.states.values():
            base_name = re.sub(r"\s*#\d+$", "", state.name)
            key = (state.android_context.activity_name, base_name)
            groups[key].append(state.state_id)

        merged_count = 0
        _MEANINGFUL_SELF_LOOPS = {"scroll_up", "scroll_down", "input_text"}  # noqa: N806

        def _fp(state_id: str) -> str:
            return fsm.states[state_id].identity.functional_hash or ""

        for (_activity, base_name), state_ids in groups.items():
            if len(state_ids) <= 1:
                continue

            # Partition into compatibility clusters by structural fingerprint.
            # States with identical fingerprints are presumed merge-safe;
            # states with distinct fingerprints under the same name are kept
            # apart and reported as diagnostics.
            clusters: dict[str, list[str]] = defaultdict(list)
            for sid in state_ids:
                clusters[_fp(sid)].append(sid)

            if len(clusters) > 1:
                logger.warning(
                    f"Skipping merge for '{base_name}' in activity {_activity}: "
                    f"{len(clusters)} incompatible structural fingerprints "
                    f"({sum(len(v) for v in clusters.values())} states)"
                )

            for fp_key, cluster_ids in clusters.items():
                if len(cluster_ids) <= 1:
                    continue
                canonical_id = cluster_ids[0]
                duplicates = cluster_ids[1:]

                for dup_id in duplicates:
                    dup_state = fsm.states[dup_id]
                    fsm.states[canonical_id].evidence.raw_screen_ids.extend(
                        dup_state.evidence.raw_screen_ids
                    )

                fsm.states[canonical_id].name = base_name

                redirect_map = {dup_id: canonical_id for dup_id in duplicates}
                new_transitions: list[Transition] = []
                seen_keys: set[tuple[Any, ...]] = set()

                for t in fsm.transitions:
                    source = redirect_map.get(t.source, t.source)
                    target = redirect_map.get(t.target, t.target)
                    if source == target:
                        atype = (t.action.get("type") or t.action.get("action_type") or "").lower()
                        # Keep meaningful self-loops (scroll / input / toggle)
                        # and observed click self-loops that carry stable
                        # element identity (Settings list rows, switches).
                        is_identity_click = atype == "click" and _click_action_has_identity(
                            t.action
                        )
                        is_observed = bool(t.provenance) or t.observed_count > 0
                        keep_self_loop = atype in _MEANINGFUL_SELF_LOOPS or (
                            is_identity_click and is_observed
                        )
                        if not keep_self_loop:
                            # Drop synthetic / unidentified click no-op self-loops.
                            continue
                    key = transition_identity_key(t, source=source, target=target)
                    if key in seen_keys:
                        for existing in new_transitions:
                            if transition_identity_key(existing) == key:
                                self._merge_transition_evidence(existing, t)
                                break
                    else:
                        seen_keys.add(key)
                        t.source = source
                        t.target = target
                        new_transitions.append(t)

                for dup_id in duplicates:
                    if dup_id in fsm.states:
                        del fsm.states[dup_id]
                    if dup_id in fsm.graph:
                        fsm.graph.remove_node(dup_id)

                fsm.graph.remove_edges_from(list(fsm.graph.edges))
                fsm.transitions = new_transitions
                for t in new_transitions:
                    if t.source in fsm.graph and t.target in fsm.graph:
                        fsm.graph.add_edge(t.source, t.target)

                if fsm.initial_state in redirect_map:
                    fsm.initial_state = redirect_map[fsm.initial_state]

                merged_count += len(duplicates)
                logger.debug(
                    f"Merged {len(duplicates)} duplicates of '{base_name}' "
                    f"(fp={fp_key[:6]}) into {canonical_id}"
                )

        return merged_count

    def _remove_error_states(self, fsm: AppFSM) -> int:
        """Remove transient error/system states from the FSM."""
        from vigil.core.platform_priors import get_error_patterns

        patterns = get_error_patterns()
        to_remove: list[str] = []
        for state in fsm.states.values():
            name_lower = state.name.lower()
            for pattern in patterns:
                if pattern.lower() in name_lower:
                    to_remove.append(state.state_id)
                    break

        for sid in to_remove:
            # Remove transitions involving this state
            fsm.transitions = [t for t in fsm.transitions if t.source != sid and t.target != sid]
            # Remove from graph
            if sid in fsm.graph:
                fsm.graph.remove_node(sid)
            # Remove from states dict
            del fsm.states[sid]
            # Update initial_state if needed
            if fsm.initial_state == sid:
                fsm.initial_state = None

        if to_remove:
            logger.debug(f"Removed error states: {to_remove}")

        return len(to_remove)

    # --- Behavioral coarsening (post-build abstraction layer) ---

    _COARSEN_MEANINGFUL_SELF_LOOPS = frozenset({"scroll_up", "scroll_down", "input_text"})

    @classmethod
    def _action_kept_self_loop(cls, action: dict[str, Any]) -> bool:
        """A self-loop survives lifting iff the action is a meaningful
        affordance (scroll/input) or an identity-bearing click."""
        atype = (action.get("type") or action.get("action_type") or "").lower()
        if atype in cls._COARSEN_MEANINGFUL_SELF_LOOPS:
            return True
        return atype == "click" and _click_action_has_identity(action)

    def _coarsen_behavioral_duplicates(
        self,
        fsm: AppFSM,
        raw_screens: dict[str, Any],
    ) -> int:
        """Compute a verifier-preserving behavioral quotient and rewrite
        the FSM to one representative per equivalence class.

        Uses fixed-point partition refinement (see
        :mod:`vigil.neuro.behavioral_quotient`):

          1. Initial partition by ``compute_quotient_label`` on each
             state's representative raw screen (volatile text, per-row
             content, EditText literals, and bounds excluded).
          2. Refinement: split a block whenever its members disagree on
             ``{ (canonical_action_key(a), block_id(target)) for outgoing
             attributable edges }``. Iterate until fixed point.
          3. Lifting: pick a representative per block (the initial state
             when present, otherwise lowest deterministic id), absorb
             every absorbed state's ``evidence.raw_screen_ids``, and
             rewrite every transition through the redirect map.
          4. Concrete-selector preservation: distinct
             ``canonical_action_key`` variants are kept as separate
             ``Transition`` rows so the runtime can locate the concrete
             selector.
          5. Canonical-action determinism guard: after lifting, any
             ``(source, canonical_action_key)`` still mapping to multiple
             distinct target blocks among attributable transitions triggers
             a defensive re-split of the offending source block.

        The refinement is fully driven by transition *block* targets, so
        thread variants A, B that send → A', B' (themselves block-equal)
        no longer cause a spurious "conflicting raw target" rejection —
        they coalesce by construction.

        Returns:
            Number of states absorbed away (i.e., ``initial_states -
            len(blocks)``).
        """
        from vigil.neuro.behavioral_quotient import (
            QuotientResult,
            TransitionRow,
            quotient_states,
        )

        if not fsm.states:
            return 0

        screen_elements_cache: dict[str, list[dict[str, Any]]] = {}
        for sid, screen in raw_screens.items():
            screen_elements_cache[sid] = screen.get(
                "interactable_elements", screen.get("elements", [])
            )

        def _representative_screen(state: AbstractState) -> dict[str, Any] | None:
            for sid in state.evidence.raw_screen_ids:
                screen = raw_screens.get(sid)
                if isinstance(screen, dict) and (
                    screen.get("elements") or screen.get("interactable_elements")
                ):
                    return screen
            return None

        # ---- 1. Per-state label, with the dialog flag taken from the
        # state-level detector so the FSM-side dialog partition stays
        # consistent with downstream `_detect_dialog_states`.
        label_cache: dict[str, dict[str, Any]] = {}
        for state_id, state in fsm.states.items():
            screen = _representative_screen(state)
            if screen is None:
                # No raw screen evidence — label by activity alone so the
                # state lands in its own (singleton) block rather than
                # accidentally merging with a labelled block.
                label_cache[state_id] = {
                    "_no_screen": True,
                    "state_id": state_id,
                    "activity": state.android_context.activity_name or "",
                }
                continue
            dialog_flag = self._is_dialog_state(state, raw_screens, screen_elements_cache)
            label_cache[state_id] = compute_quotient_label(screen, dialog_flag=dialog_flag)
            state.identity.canonical_equivalence_hash = signature_hash(label_cache[state_id])

        def _label_fn(state_id: str) -> dict[str, Any]:
            return label_cache.get(state_id, {"state_id": state_id})

        rows = [
            TransitionRow(
                source=t.source,
                target=t.target,
                action=t.action,
                ambiguous_evidence=t.ambiguous_evidence,
            )
            for t in fsm.transitions
        ]

        result: QuotientResult = quotient_states(
            states=list(fsm.states.keys()),
            transitions=rows,
            label_fn=_label_fn,
            initial_state=fsm.initial_state,
        )
        transition_action_key_by_index = [canonical_action_key(row.action) for row in rows]

        # ---- 2. Canonical-action determinism guard. The refinement step
        # already partitions by block-level successors, but edges marked
        # ambiguous_evidence were excluded from refinement and the lifted
        # result must still satisfy the verifier invariant
        # "(source_block, canonical_action_key) -> at most one target
        # block among attributable edges". For every conflicting
        # ``(source_block, canonical_action_key)``, the offending source
        # block is split by ``tgt_block`` for that specific action key
        # (members that didn't observe it stay in the largest piece).
        # The redirect is rebuilt before being applied to the live FSM,
        # so the verifier invariant is never violated by a behavioral
        # merge; an unsplittable residual conflict raises instead of
        # proceeding. A ``builder_diagnostic`` evolution-log entry
        # records what was split for downstream comparators.
        redirect = result.redirect_map()
        for det_pass in range(_MAX_DETERMINISM_PASSES + 1):
            action_key_targets_final: dict[tuple[str, Hashable], set[str]] = defaultdict(set)
            action_key_state_to_tgt: dict[tuple[str, Hashable], dict[str, str]] = defaultdict(dict)
            for idx, t in enumerate(fsm.transitions):
                if t.ambiguous_evidence:
                    continue
                src_p = redirect.get(t.source, t.source)
                tgt_p = redirect.get(t.target, t.target)
                if src_p == tgt_p and not self._action_kept_self_loop(t.action):
                    continue
                action_key = transition_action_key_by_index[idx]
                src_block = result.state_to_block.get(t.source) or src_p
                tgt_block = result.state_to_block.get(t.target) or tgt_p
                action_key_targets_final[(src_block, action_key)].add(tgt_block)
                action_key_state_to_tgt[(src_block, action_key)].setdefault(t.source, tgt_block)
            residual_conflicts = [
                (src_block, action_key, tgts)
                for (src_block, action_key), tgts in action_key_targets_final.items()
                if len(tgts) > 1
            ]
            if not residual_conflicts:
                break
            if det_pass == _MAX_DETERMINISM_PASSES:
                src_block, action_key, tgts = residual_conflicts[0]
                raise RuntimeError(
                    "Behavioral-quotient determinism invariant violated: "
                    f"source_block={src_block} action_key={action_key} "
                    f"targets={sorted(tgts)}"
                )
            for src_block, action_key, tgts in residual_conflicts:
                members = set(result.block_to_members.get(src_block, set()))
                if len(members) <= 1:
                    continue
                state_to_tgt = action_key_state_to_tgt.get((src_block, action_key), {})
                groups: dict[str, list[str]] = defaultdict(list)
                no_obs: list[str] = []
                for sid in members:
                    tb = state_to_tgt.get(sid)
                    if tb is None:
                        no_obs.append(sid)
                    else:
                        groups[tb].append(sid)
                if len(groups) <= 1:
                    continue
                ordered = sorted(
                    groups.items(),
                    key=lambda kv: (-len(kv[1]), sorted(kv[1])[0]),
                )
                _, keep_states = ordered[0]
                keep_states = sorted(set(keep_states) | set(no_obs))
                result.block_to_members[src_block] = set(keep_states)
                if fsm.initial_state in keep_states:
                    result.block_to_representative[src_block] = fsm.initial_state
                else:
                    result.block_to_representative[src_block] = min(keep_states)
                for _tb, sub_states in ordered[1:]:
                    new_block_id = f"{src_block}__safety_{len(result.block_to_members)}"
                    result.block_to_members[new_block_id] = set(sub_states)
                    result.block_label_hash[new_block_id] = result.block_label_hash.get(
                        src_block, ""
                    )
                    for sid in sub_states:
                        result.state_to_block[sid] = new_block_id
                    if fsm.initial_state in sub_states:
                        result.block_to_representative[new_block_id] = fsm.initial_state
                    else:
                        result.block_to_representative[new_block_id] = min(sub_states)
                result.construction_log_entries.append(
                    {
                        "action": "builder_diagnostic",
                        "kind": "quotient_residual_conflict",
                        "source_block": src_block,
                        "canonical_action_key": (
                            list(action_key) if isinstance(action_key, tuple) else action_key
                        ),
                        "distinct_target_blocks": sorted(tgts),
                        "split_groups": {tb: sorted(states) for tb, states in ordered},
                    }
                )
            redirect = result.redirect_map()

        # ---- 3. Apply the redirect map to the live FSM. Concrete-selector
        # variants survive because dedup is keyed on canonical_action_key.
        absorbed_total = self._apply_redirect_map(
            fsm,
            redirect,
            construction_entries=result.construction_log_entries,
        )
        if absorbed_total:
            logger.debug(
                "Behavioral quotient: {} blocks (was {} states), {} passes",
                len(result.block_to_members),
                len(fsm.states) + absorbed_total,
                result.refinement_passes,
            )
        return absorbed_total

    def _apply_redirect_map(
        self,
        fsm: AppFSM,
        redirect: dict[str, str],
        *,
        construction_entries: list[dict[str, Any]] | None = None,
    ) -> int:
        """Apply a state-id redirect map to ``fsm``.

        Absorbs ``evidence.raw_screen_ids`` from absorbed states onto
        their representatives, rewrites every transition through the
        redirect (dropping meaningless self-loops, deduping by
        ``(source, target, canonical_action_key)`` via
        :meth:`_merge_transition_evidence`), deletes absorbed states,
        rebuilds the graph, and follows ``fsm.initial_state`` through
        the redirect. Returns the number of states absorbed away.

        Identity entries (``redirect[s] == s``) are no-ops. Used by
        :meth:`_coarsen_behavioral_duplicates`; the helper exists so the
        rewire is testable in isolation.
        """
        absorbed_total: set[str] = {
            s for s, rep in redirect.items() if s != rep and rep in fsm.states
        }
        if not absorbed_total:
            return 0

        # Absorb raw_screen_ids.
        rep_to_absorbed: dict[str, list[str]] = defaultdict(list)
        for s, rep in redirect.items():
            if s == rep:
                continue
            if rep not in fsm.states:
                continue
            rep_to_absorbed[rep].append(s)
        for rep, absorbed in rep_to_absorbed.items():
            rep_state = fsm.states[rep]
            rep_screens = set(rep_state.evidence.raw_screen_ids)
            for abs_id in absorbed:
                abs_state = fsm.states.get(abs_id)
                if abs_state is None:
                    continue
                for raw_sid in abs_state.evidence.raw_screen_ids:
                    if raw_sid not in rep_screens:
                        rep_state.evidence.raw_screen_ids.append(raw_sid)
                        rep_screens.add(raw_sid)

        # Rewrite transitions.
        new_transitions: list[Transition] = []
        seen_keys: dict[tuple[Any, ...], Transition] = {}
        for t in fsm.transitions:
            src = redirect.get(t.source, t.source)
            tgt = redirect.get(t.target, t.target)
            if src == tgt and not self._action_kept_self_loop(t.action):
                continue
            key = transition_identity_key(t, source=src, target=tgt)
            if key in seen_keys:
                self._merge_transition_evidence(seen_keys[key], t)
                continue
            t.source = src
            t.target = tgt
            seen_keys[key] = t
            new_transitions.append(t)

        # Delete absorbed states.
        for abs_id in absorbed_total:
            if abs_id in fsm.states:
                del fsm.states[abs_id]
            if abs_id in fsm.graph:
                fsm.graph.remove_node(abs_id)

        fsm.transitions = new_transitions
        fsm.graph.remove_edges_from(list(fsm.graph.edges))
        for t in new_transitions:
            if t.source in fsm.graph and t.target in fsm.graph:
                fsm.graph.add_edge(t.source, t.target)

        if fsm.initial_state in absorbed_total:
            fsm.initial_state = redirect[fsm.initial_state]

        if construction_entries:
            fsm.construction_log.extend(construction_entries)
        return len(absorbed_total)

    def _detect_dialog_states(
        self,
        fsm: AppFSM,
        raw_screens: dict[str, Any],
    ) -> int:
        """Detect dialog states and set hierarchy_level=COMPONENT with parent."""
        detected = 0

        screen_elements_cache: dict[str, list[dict[str, Any]]] = {}
        for sid, screen in raw_screens.items():
            screen_elements_cache[sid] = screen.get(
                "interactable_elements", screen.get("elements", [])
            )

        for state_id, state in fsm.states.items():
            if self._is_dialog_state(state, raw_screens, screen_elements_cache):
                state.hierarchy_level = HierarchyLevel.COMPONENT

                parent_id = self._find_dialog_parent(state_id, fsm)
                if parent_id:
                    state.parent_state = parent_id

                detected += 1

        return detected

    def _is_dialog_state(
        self,
        state: AbstractState,
        raw_screens: dict[str, Any],
        screen_elements_cache: dict[str, list[dict[str, Any]]],
    ) -> bool:
        """Check if a state represents a dialog/picker overlay."""
        from vigil.core.platform_priors import get_dialog_indicators

        indicators = get_dialog_indicators()
        dialog_classes = set(indicators.get("classes", []))
        dialog_rids = {rid.lower() for rid in indicators.get("resource_ids", [])}

        for sid in state.evidence.raw_screen_ids:
            screen = raw_screens.get(sid, {})
            metadata = screen.get("metadata", {})
            if metadata.get("has_modal"):
                return True

            elements = screen_elements_cache.get(sid, [])
            rids = {(el.get("resource_id") or "").lower() for el in elements}
            if rids & dialog_rids:
                return True

            classes = {(el.get("class_name") or "").rsplit(".", 1)[-1] for el in elements}
            if classes & dialog_classes:
                return True

        return False

    @staticmethod
    def _find_dialog_parent(dialog_state_id: str, fsm: AppFSM) -> str | None:
        """Find the most likely parent state for a dialog."""
        source_counts: dict[str, int] = defaultdict(int)
        for t in fsm.transitions:
            if t.target == dialog_state_id and t.source != dialog_state_id:
                source_counts[t.source] += 1
        if not source_counts:
            return None
        return max(source_counts, key=source_counts.get)  # type: ignore[arg-type]

    def _classify_containers_structural(self, fsm: AppFSM) -> int:
        """Fallback container classification when the grounder wasn't run.

        Labels a state ``DYNAMIC`` when it (a) contains a scrollable element
        and (b) has at least two outgoing click transitions whose targets
        share the same functional fingerprint — the structural signature of
        a list-of-items page. Only runs on states still labeled ``NONE``
        so grounder decisions are preserved.

        Returns:
            Number of states newly labeled DYNAMIC.
        """
        labeled = 0

        for sid, state in fsm.states.items():
            if state.abstraction.container_type != ContainerType.NONE:
                continue

            has_scrollable = False
            for rsid in state.evidence.raw_screen_ids:
                screen = self._raw_screens.get(rsid, {})
                elements = screen.get("interactable_elements", screen.get("elements", []))
                if any(e.get("is_scrollable") for e in elements):
                    has_scrollable = True
                    break
            if not has_scrollable:
                continue

            targets = self._find_same_fingerprint_targets(fsm, sid)
            if len(targets) >= 2:
                state.abstraction.container_type = ContainerType.DYNAMIC
                labeled += 1

        return labeled

    def _build_sub_fsm_templates(self, fsm: AppFSM) -> int:
        """Create Sub-FSM templates for verified dynamic containers.

        Detects states with container_type=DYNAMIC that have multiple outgoing
        click transitions whose targets share the same ``structural_fingerprint``
        (the text-agnostic structural skeleton — the correct identity for
        "list of structurally-identical detail pages" patterns like an email
        inbox or a Wi-Fi network list).

        Collapses those N transitions into a single SubFsmTemplate reference:
        keeps one representative target state (with its descendant subgraph
        up to depth 5), removes the N-1 duplicates, redirects their incoming
        transitions to the representative, and stamps
        ``sub_fsm_template_id`` on the container state.

        Emits a diagnostic log at INFO when 0 templates are created but
        DYNAMIC containers exist — separates a data-coverage gap (few
        items observed per container) from a code bug.

        Returns:
            Number of templates created.
        """
        templates_created = 0
        dynamic_state_ids = [
            sid
            for sid, s in fsm.states.items()
            if s.abstraction.container_type == ContainerType.DYNAMIC
        ]
        # Cumulative redirect map across template iterations. A collapsed
        # state may itself be redirected by a later template; resolving
        # transitively (`_resolve_redirect`) ensures previously-built
        # templates and live transitions never reference a deleted id.
        for state_id in dynamic_state_ids:
            # The container itself may have been redirected by an earlier
            # collapse. Resolve through the cumulative map; skip if the chain
            # leads to a non-live state.
            resolved_container = self._resolve_redirect(state_id)
            if resolved_container not in fsm.states:
                continue
            state_id = resolved_container
            state = fsm.states[state_id]

            click_targets = self._find_same_fingerprint_targets(fsm, state_id)
            if len(click_targets) < 2:
                continue

            representative_target_id = click_targets[0][0]
            if representative_target_id == fsm.initial_state:
                # Defensive: never collapse the entry state away.
                continue

            # Conservative dry-run: simulate the redirect (every id in
            # ``collapsed_others`` -> ``representative_target_id``) across the
            # *entire* transition set and reject the collapse if it would
            # introduce nondeterminism — i.e. any (source, canonical_action_key)
            # mapping to more than one distinct target among attributable
            # transitions. Ambiguously attributed transitions are
            # refinement evidence; they are excluded from the check so the dry
            # run does not flag pre-existing ambiguity as a collapse failure.
            # Self-loops are not special-cased: a "self vs other-target"
            # conflict still has two distinct targets and is correctly
            # rejected. Only an actual single-target outcome (including a
            # genuine self-loop alone) is accepted.
            collapsed_others_preview = {tid for tid, _ in click_targets[1:]}

            def _rewrite_id(
                _x: str,
                _others: set[str] = collapsed_others_preview,
                _rep: str = representative_target_id,
            ) -> str:
                return _rep if _x in _others else _x

            hypothetical: dict[tuple[str, tuple[tuple[str, Any], ...]], set[str]] = defaultdict(set)
            for t in fsm.transitions:
                if t.ambiguous_evidence:
                    continue
                src_p = _rewrite_id(t.source)
                tgt_p = _rewrite_id(t.target)
                hypothetical[(src_p, canonical_action_key(t.action))].add(tgt_p)

            conflicts = [
                (src, key, sorted(tgts))
                for (src, key), tgts in hypothetical.items()
                if len(tgts) > 1
            ]
            if conflicts:
                src, key, tgts = conflicts[0]
                logger.info(
                    "Rejecting Sub-FSM template collapse for container "
                    f"{state_id}: nondeterministic post-collapse transitions "
                    f"({len(conflicts)} conflict(s)). First: source={src} "
                    f"action_key={key} targets={tgts}"
                )
                continue

            template_id = f"tmpl_{state_id}"
            rep_state = fsm.states[representative_target_id]

            # Template subgraph: representative + bounded DFS descendants
            # excluding any transition that returns to the container
            # source state. Prevents the whole FSM from being swallowed into
            # one template when navigate-back forms a back-edge.
            template_states: dict[str, AbstractState] = {representative_target_id: rep_state}
            self._collect_template_subgraph(
                fsm,
                root_id=representative_target_id,
                exclude_id=state_id,
                max_depth=_TEMPLATE_SUBGRAPH_MAX_DEPTH,
                out=template_states,
            )
            template_transitions: list[Transition] = [
                t
                for t in fsm.transitions
                if t.source in template_states and t.target in template_states
            ]

            tmpl = SubFsmTemplate(
                template_id=template_id,
                source_state_id=state_id,
                states=template_states,
                transitions=template_transitions,
            )
            fsm.sub_fsm_templates[template_id] = tmpl
            state.abstraction.template_id = template_id
            templates_created += 1

            # Lossless collapse: merge each collapsed sibling's raw_screens
            # AND its incoming/outgoing transitions onto the representative
            # before deleting it. Validator (validate_fsm) inverts
            # AbstractState.evidence.raw_screen_ids to recover screen→state
            # mappings, so any screen left orphaned by the old "delete +
            # prune" policy would surface as state_not_found on its own
            # training trace.
            #
            # Collapsed siblings share structural_hash by construction
            # (the collapse criterion in _find_same_fingerprint_targets), so
            # the representative's structural_hash already covers them;
            # we deliberately do NOT overwrite it.
            collapsed_others = {tid for tid, _ in click_targets[1:]}
            rep_screens = set(rep_state.evidence.raw_screen_ids)
            for tid in collapsed_others:
                absorbed_state = fsm.states[tid]
                for sid in absorbed_state.evidence.raw_screen_ids:
                    if sid not in rep_screens:
                        rep_state.evidence.raw_screen_ids.append(sid)
                        rep_screens.add(sid)

            # Re-source/-target transitions touching collapsed siblings onto
            # the representative, then dedupe via _merge_transition_evidence so
            # provenance and observed_count aggregate correctly.
            for t in fsm.transitions:
                if t.source in collapsed_others:
                    t.source = representative_target_id
                if t.target in collapsed_others:
                    t.target = representative_target_id

            merged_transitions: dict[tuple[Any, ...], Transition] = {}
            for t in fsm.transitions:
                key = transition_identity_key(t)
                if key in merged_transitions:
                    self._merge_transition_evidence(merged_transitions[key], t)
                else:
                    merged_transitions[key] = t
            fsm.transitions = list(merged_transitions.values())
            self._downgrade_post_collapse_conflicts(representative_target_id, fsm.transitions)

            for tid in collapsed_others:
                del fsm.states[tid]
                if tid in fsm.graph:
                    fsm.graph.remove_node(tid)
                # Insert the literal representative (live by construction at
                # the start of this iteration). Walking _resolve_redirect here
                # could chain through an older entry pointing back at a state
                # also in collapsed_others, producing a self-loop. Transitive
                # resolution from prior keys still works because _resolve_redirect
                # walks the chain when older lookups land on this entry.
                self._collapse_redirects[tid] = representative_target_id

            # Update previously-created SubFsmTemplates so none of them
            # reference a state that was just deleted. Each tmpl's
            # source_state_id / states / transitions are rewritten through
            # the cumulative redirect map.
            for prior in fsm.sub_fsm_templates.values():
                self._rewrite_template_through_redirects(prior, fsm)

            fsm.graph.remove_edges_from(list(fsm.graph.edges))
            for t in fsm.transitions:
                if t.source in fsm.graph and t.target in fsm.graph:
                    fsm.graph.add_edge(t.source, t.target)

            logger.debug(
                f"Template {template_id}: collapsed {len(click_targets)} "
                f"transitions from {state_id} (kept {representative_target_id})"
            )

        if templates_created == 0 and dynamic_state_ids:
            # Diagnostic: show per-container click-target structural_fp
            # distribution so the operator can tell coverage gap from a
            # matching bug.
            logger.info(
                f"No sub-FSM templates created despite "
                f"{len(dynamic_state_ids)} DYNAMIC container(s). "
                "Per-container click-target structural fingerprint counts:"
            )
            for sid in dynamic_state_ids:
                counts: dict[str, int] = defaultdict(int)
                for t in fsm.transitions:
                    if t.source != sid or t.action.get("type") != "click":
                        continue
                    tgt = fsm.states.get(t.target)
                    if tgt is None:
                        continue
                    sfp = tgt.identity.structural_hash or "<none>"
                    counts[sfp] += 1
                summary = ", ".join(f"{k[:8]}:{v}" for k, v in counts.items())
                logger.info(f"  {sid[:6]}  {summary or '<no click targets>'}")

        # Stale template-id sweep: any state whose container_type is not
        # DYNAMIC (perhaps because the state was absorbed/redirected, or
        # the classifier downgraded it) must not retain a template_id.
        # Without this, ``resolve_transition`` would treat clicks on a
        # non-dynamic state as template-binding attempts.
        for state in fsm.states.values():
            if (
                state.abstraction.template_id is not None
                and state.abstraction.container_type != ContainerType.DYNAMIC
            ):
                state.abstraction.template_id = None

        # Drop orphan templates that no live state references via
        # ``abstraction.template_id``. After redirects, a previously-built
        # template may no longer be reachable from any state and should
        # not linger in the bundle.
        referenced_template_ids = {
            s.abstraction.template_id
            for s in fsm.states.values()
            if s.abstraction.template_id is not None
        }
        orphan_template_ids = [
            tid for tid in list(fsm.sub_fsm_templates.keys()) if tid not in referenced_template_ids
        ]
        for tid in orphan_template_ids:
            del fsm.sub_fsm_templates[tid]
        if orphan_template_ids:
            logger.debug(f"Removed {len(orphan_template_ids)} orphan Sub-FSM template(s)")

        return templates_created

    def _resolve_redirect(self, state_id: str) -> str:
        """Follow ``self._collapse_redirects`` transitively to find the live
        representative for ``state_id``. The ``seen`` set terminates the walk
        on any accidental cycle in the redirect chain."""
        seen: set[str] = set()
        current = state_id
        while current not in seen:
            seen.add(current)
            nxt = self._collapse_redirects.get(current)
            if nxt is None or nxt == current:
                return current
            current = nxt
        return current

    def _rewrite_template_through_redirects(self, tmpl: SubFsmTemplate, fsm: AppFSM) -> None:
        """Rewrite a SubFsmTemplate so every state id it references is alive.

        Resolves ``source_state_id``, every key in ``states``, and every
        ``(source, target)`` in ``transitions`` through the cumulative redirect
        map. Entries that resolve to a non-live state are dropped. ``states``
        values are replaced with the live ``fsm.states[id]`` so absorbed
        ``raw_screens`` propagate into the template view. Transitions are
        deduped by ``(source, target, canonical_action_key)`` post-rewrite.
        """
        live_state_ids = set(fsm.states.keys())

        new_source = self._resolve_redirect(tmpl.source_state_id)
        if new_source in live_state_ids:
            tmpl.source_state_id = new_source

        rewritten_states: dict[str, AbstractState] = {}
        for sid in list(tmpl.states.keys()):
            resolved = self._resolve_redirect(sid)
            if resolved in live_state_ids and resolved not in rewritten_states:
                rewritten_states[resolved] = fsm.states[resolved]
        tmpl.states.clear()
        tmpl.states.update(rewritten_states)

        merged: dict[tuple[Any, ...], Transition] = {}
        for t in tmpl.transitions:
            new_src = self._resolve_redirect(t.source)
            new_tgt = self._resolve_redirect(t.target)
            if new_src not in live_state_ids or new_tgt not in live_state_ids:
                continue
            t.source = new_src
            t.target = new_tgt
            key = transition_identity_key(t)
            if key in merged:
                self._merge_transition_evidence(merged[key], t)
            else:
                merged[key] = t
        tmpl.transitions = list(merged.values())

    def _downgrade_post_collapse_conflicts(
        self, representative_state_id: str, transitions: list[Transition]
    ) -> None:
        """Downgrade conflicts introduced by template sibling collapse."""
        groups: dict[tuple[str, tuple[tuple[str, Any], ...]], list[Transition]] = defaultdict(list)
        for t in transitions:
            if t.source != representative_state_id:
                continue
            groups[(t.source, canonical_action_key(t.action))].append(t)

        for (source_id, action_key), conflicting in groups.items():
            if len({t.target for t in conflicting}) <= 1:
                continue
            self._downgrade_conflict(
                conflicting,
                source_id,
                action_key,
                "post_collapse_conflict",
            )

    @staticmethod
    def _collect_template_subgraph(
        fsm: AppFSM,
        *,
        root_id: str,
        exclude_id: str,
        max_depth: int,
        out: dict[str, AbstractState],
    ) -> None:
        """DFS from ``root_id`` up to ``max_depth`` hops, skipping any
        transition whose target is ``exclude_id`` (the container we came
        from — a navigate-back edge would otherwise swallow the whole FSM).
        Populates ``out`` in place."""
        stack: list[tuple[str, int]] = [(root_id, 0)]
        while stack:
            sid, depth = stack.pop()
            if depth >= max_depth:
                continue
            for t in fsm.transitions:
                if t.source != sid:
                    continue
                if t.target == exclude_id or t.target == sid:
                    continue
                child = fsm.states.get(t.target)
                if child is None or t.target in out:
                    continue
                out[t.target] = child
                stack.append((t.target, depth + 1))

    @staticmethod
    def _find_same_fingerprint_targets(fsm: AppFSM, source_id: str) -> list[tuple[str, str]]:
        """Find click transitions from ``source_id`` whose targets share a
        ``structural_fingerprint`` (the text-agnostic structural skeleton).

        Grouping on ``structural_fingerprint`` rather than the text-anchored
        ``fingerprint`` is the correct use-case for list-like containers:
        every row clicks into a structurally-identical detail page but with
        distinct row text, so text-anchored identities diverge while the
        structural one collapses them.

        Returns:
            List of (target_state_id, structural_fingerprint) for the
            largest group of same-structural-fingerprint targets. Empty if
            no group has >= 2 members.
        """
        fp_groups: dict[str, list[str]] = defaultdict(list)
        seen_per_group: dict[str, set[str]] = defaultdict(set)
        for t in fsm.transitions:
            if t.source != source_id:
                continue
            if t.action.get("type") != "click":
                continue
            target = fsm.states.get(t.target)
            if target is None:
                continue
            sfp = target.identity.structural_hash
            if not sfp:
                continue
            # Multiple observations of the same (source, target) edge must not
            # produce duplicate group members; otherwise the collapse loop can
            # treat the representative as one of its own collapsed_others.
            if t.target in seen_per_group[sfp]:
                continue
            seen_per_group[sfp].add(t.target)
            fp_groups[sfp].append(t.target)

        best_group: list[str] = []
        best_fp = ""
        for fp, targets in fp_groups.items():
            if len(targets) > len(best_group):
                best_group = targets
                best_fp = fp

        if len(best_group) < 2:
            return []
        return [(tid, best_fp) for tid in best_group]

    # --- APE-style refinement ---

    def _refine_conflicting_successors(
        self,
        states: dict[str, AbstractState],
        sid_to_state_id: dict[str, str],
        transitions: list[Transition],
        raw_screens: dict[str, Any],
    ) -> list[Transition]:
        """Split or downgrade states where the same canonical action yields
        conflicting successors (the APE-style refinement step).

        After ``_merge_transitions``, two outgoing edges of the same source
        with the same ``canonical_action_key`` but different targets prove the
        abstract state actually represents two distinct concrete situations.
        We try to split, but only conservatively:

          1. Every conflicting edge must carry provenance pointing at a real
             ``source_screen_id``; otherwise we cannot attribute the conflict
             to specific raw screens and we downgrade.
          2. The raw source screens that lead to each target must be
             distinguishable by stable *secondary* features (activity,
             modal flag, non-editable text anchors, content descriptions)
             that are NOT already captured by the primary fingerprint. If the
             screens are indistinguishable, we keep both edges and mark them
             ``ambiguous_evidence=True``.

        Diagnostics are buffered on the builder as ``self._refinement_log``;
        the caller copies them into ``AppFSM.construction_log`` after the FSM is
        assembled (this method runs before the AppFSM exists).
        """
        self._refinement_log = []

        conflict_groups: dict[tuple[str, tuple[tuple[str, Any], ...]], list[Transition]] = (
            defaultdict(list)
        )
        for t in transitions:
            conflict_groups[(t.source, canonical_action_key(t.action))].append(t)

        working = list(transitions)

        for (source_id, action_key), conflicting in conflict_groups.items():
            distinct_targets = {t.target for t in conflicting}
            if len(distinct_targets) <= 1:
                continue
            source_state = states.get(source_id)
            if source_state is None:
                continue

            target_to_screens: dict[str, set[str]] = defaultdict(set)
            attribution_complete = True
            for t in conflicting:
                screens_for_t = {p.source_screen_id for p in t.provenance if p.source_screen_id}
                if not screens_for_t:
                    attribution_complete = False
                    break
                target_to_screens[t.target].update(screens_for_t)

            if not attribution_complete:
                self._downgrade_conflict(conflicting, source_id, action_key, "missing_provenance")
                continue

            sig_to_target: dict[Any, str] = {}
            ambiguous = False
            for tgt, screens in target_to_screens.items():
                for sid in screens:
                    sig = secondary_feature_signature(raw_screens.get(sid, {}))
                    seen = sig_to_target.get(sig)
                    if seen is not None and seen != tgt:
                        ambiguous = True
                        break
                    sig_to_target[sig] = tgt
                if ambiguous:
                    break

            if ambiguous:
                self._downgrade_conflict(
                    conflicting, source_id, action_key, "indistinguishable_secondary_features"
                )
                continue

            # Stricter precondition for splitting: every target's raw screens must
            # produce a SINGLE stable secondary-feature signature. If a target's
            # screens span multiple signatures, the resulting refined sibling
            # would need a group-hash fingerprint that StateLocator cannot safely
            # disambiguate from a serialized FSM alone. Downgrade instead so the
            # validator surfaces the ambiguity without a malformed split.
            multi_signature_target = False
            for screens in target_to_screens.values():
                distinct_sigs = {
                    secondary_feature_signature(raw_screens.get(sid, {})) for sid in screens
                }
                if len(distinct_sigs) > 1:
                    multi_signature_target = True
                    break
            if multi_signature_target:
                self._downgrade_conflict(
                    conflicting,
                    source_id,
                    action_key,
                    "target_spans_multiple_secondary_signatures",
                )
                continue

            working = self._split_state_for_conflict(
                states=states,
                sid_to_state_id=sid_to_state_id,
                transitions=working,
                raw_screens=raw_screens,
                source_state=source_state,
                conflicting=conflicting,
                target_to_screens=target_to_screens,
                action_key=action_key,
            )

        return working

    def _downgrade_conflict(
        self,
        conflicting: list[Transition],
        source_id: str,
        action_key: tuple[tuple[str, Any], ...],
        reason: str,
    ) -> None:
        """Mark an unresolvable determinism conflict as unattributable evidence.

        Every conflicting edge stays in the transition graph -- each is real observed
        behaviour. The marking keeps offline refinement from partitioning on
        evidence that cannot say which source screen leads where. At runtime no
        flag is consulted: the edges share a source and canonical action key, so
        :meth:`AppFSM.resolve_transition` already reports ambiguity with
        "Multiple transitions share the same canonical action key", which is the
        paper's incomplete-action-resolution clause.
        """
        for t in conflicting:
            t.ambiguous_evidence = True
        self._refinement_log.append(
            {
                "action": "downgrade",
                "source_state": source_id,
                "canonical_action_key": [list(item) for item in action_key],
                "targets": sorted({t.target for t in conflicting}),
                "reason": reason,
            }
        )

    def _split_state_for_conflict(
        self,
        *,
        states: dict[str, AbstractState],
        sid_to_state_id: dict[str, str],
        transitions: list[Transition],
        raw_screens: dict[str, Any],
        source_state: AbstractState,
        conflicting: list[Transition],
        target_to_screens: dict[str, set[str]],
        action_key: tuple[tuple[str, Any], ...],
    ) -> list[Transition]:
        """Split ``source_state`` so that each conflicting target gets its own
        specialized source state. Keeps the first target on the original state
        and creates ``s_xxx__refined_N`` siblings for the rest.

        The split must keep three things in sync:
          - ``states`` (new entries added)
          - ``sid_to_state_id`` (raw screens re-attributed)
          - ``transitions`` (conflicting edges re-sourced, other outgoing
            edges replicated, incoming edges re-targeted via provenance)
        """
        source_id = source_state.state_id
        target_order = sorted(target_to_screens.keys())
        keep_target = target_order[0]
        new_state_ids: dict[str, str] = {}
        base_fingerprint = source_state.identity.functional_hash
        base_structural_fingerprint = source_state.identity.structural_hash
        target_secondary_hashes = {
            tgt: secondary_feature_group_hash(screens, raw_screens)
            for tgt, screens in target_to_screens.items()
        }

        keep_hash = target_secondary_hashes[keep_target]
        source_state.identity.functional_hash = with_secondary_feature_hash(
            base_fingerprint, keep_hash
        )
        source_state.identity.structural_hash = with_secondary_feature_hash(
            base_structural_fingerprint, keep_hash
        )

        for idx, tgt in enumerate(target_order[1:], start=1):
            base = f"{source_id}__refined_{idx}"
            new_state_id = base
            collision = 1
            while new_state_id in states:
                collision += 1
                new_state_id = f"{base}_{collision}"
            screens_for_new = target_to_screens[tgt]
            secondary_hash = target_secondary_hashes[tgt]
            new_state = AbstractState(
                state_id=new_state_id,
                name=f"{source_state.name} #refined-{idx}",
                hierarchy_level=source_state.hierarchy_level,
                parent_state=source_state.parent_state,
                identity=StateIdentity(
                    functional_hash=with_secondary_feature_hash(base_fingerprint, secondary_hash),
                    structural_hash=with_secondary_feature_hash(
                        base_structural_fingerprint, secondary_hash
                    ),
                ),
                android_context=AndroidStateContext(
                    activity_name=source_state.android_context.activity_name,
                    package_name=source_state.android_context.package_name,
                    window_type=source_state.android_context.window_type,
                ),
                evidence=StateEvidence(raw_screen_ids=sorted(screens_for_new)),
                abstraction=StateAbstraction(
                    container_type=source_state.abstraction.container_type,
                    container_selector=dict(source_state.abstraction.container_selector),
                    template_id=source_state.abstraction.template_id,
                    parameter_schema=dict(source_state.abstraction.parameter_schema),
                ),
                annotations=source_state.annotations.model_copy(deep=True),
                invariant_specs=[
                    spec.model_copy(deep=True) for spec in source_state.invariant_specs
                ],
            )
            states[new_state_id] = new_state
            new_state_ids[tgt] = new_state_id
            for sid in screens_for_new:
                sid_to_state_id[sid] = new_state_id

        relocated_screens = {s for tgt in target_order[1:] for s in target_to_screens[tgt]}
        source_state.evidence.raw_screen_ids = [
            s for s in source_state.evidence.raw_screen_ids if s not in relocated_screens
        ]

        # 1. Update conflicting transitions: re-source the ones whose target is not the kept one.
        for t in conflicting:
            if t.target == keep_target:
                continue
            new_src = new_state_ids.get(t.target)
            if new_src:
                t.source = new_src

        # 2. Partition other outgoing transitions by provenance source screen.
        target_to_state_id = {keep_target: source_id, **new_state_ids}
        state_to_raw_screens = {
            target_to_state_id[tgt]: set(screens) for tgt, screens in target_to_screens.items()
        }
        other_outgoing = [t for t in transitions if t.source == source_id and t not in conflicting]
        rebuilt_outgoing: list[Transition] = []
        for t in other_outgoing:
            source_screen_ids = {p.source_screen_id for p in t.provenance if p.source_screen_id}
            matched_states = [
                state_id
                for state_id, screens in state_to_raw_screens.items()
                if source_screen_ids & screens
            ]
            unattributed = not source_screen_ids
            destinations = list(state_to_raw_screens) if unattributed else matched_states
            spans_multiple_siblings = len(matched_states) > 1

            for state_id in destinations:
                if unattributed:
                    provenance = list(t.provenance)
                else:
                    provenance = [
                        p
                        for p in t.provenance
                        if p.source_screen_id in state_to_raw_screens[state_id]
                    ]
                ambiguous_partition = unattributed or spans_multiple_siblings
                # Keep unattributed/broad evidence usable without letting it drive a
                # partition decision on every sibling.
                rebuilt_outgoing.append(
                    t.model_copy(
                        deep=True,
                        update={
                            "source": state_id,
                            "ambiguous_evidence": (t.ambiguous_evidence or ambiguous_partition),
                            "observed_count": (len(provenance) if provenance else t.observed_count),
                            "provenance": provenance,
                        },
                    )
                )

        transitions = [t for t in transitions if t not in other_outgoing] + rebuilt_outgoing

        # 3. Redirect incoming transitions based on provenance.target_screen_id.
        for t in transitions:
            if t.target != source_id or not t.provenance:
                continue
            redirects = set()
            for entry in t.provenance:
                if entry.target_screen_id and entry.target_screen_id in sid_to_state_id:
                    redirects.add(sid_to_state_id[entry.target_screen_id])
            redirects.discard(source_id)
            if len(redirects) == 1:
                t.target = next(iter(redirects))

        self._refinement_log.append(
            {
                "action": "split",
                "source_state": source_id,
                "new_states": list(new_state_ids.values()),
                "canonical_action_key": [list(item) for item in action_key],
                "targets": target_order,
            }
        )
        return transitions

    def _build_states(
        self,
        raw_screens: dict[str, Any],
        trace_dir: Path | None = None,
        app_prior: AppPrior | None = None,
    ) -> tuple[dict[str, str], dict[str, AbstractState]]:
        """Build AbstractStates from screens, one per unique identity.

        A screen's identity is its explorer-computed ``state_id`` (a
        functional, text-anchored hybrid key) when present, otherwise the
        scroll-aware functional fingerprint computed here.

        Returns:
            (screen_id -> canonical_state_id, canonical_state_id -> AbstractState).
        """
        sid_to_state_id: dict[str, str] = {}
        states: dict[str, AbstractState] = {}
        identity_to_canonical: dict[str, str] = {}
        state_counter = 0

        for screen_id, screen in raw_screens.items():
            identity: str = screen.get("state_id") or self._compute_functional_fingerprint(screen)
            if not identity:
                continue
            canonical = identity_to_canonical.get(identity)
            if canonical is None:
                state_counter += 1
                canonical = f"s_{state_counter:03d}"
                identity_to_canonical[identity] = canonical
                structural_fp = self._compute_structural_fingerprint(screen)
                states[canonical] = AbstractState(
                    state_id=canonical,
                    name=self._derive_state_name(screen, canonical, trace_dir, app_prior),
                    hierarchy_level=HierarchyLevel.ACTIVITY,
                    identity=StateIdentity(
                        functional_hash=identity,
                        structural_hash=structural_fp or None,
                    ),
                    android_context=AndroidStateContext(activity_name=screen.get("activity_name")),
                    evidence=StateEvidence(raw_screen_ids=[screen_id]),
                )
            else:
                states[canonical].evidence.raw_screen_ids.append(screen_id)
            sid_to_state_id[screen_id] = canonical

        logger.info(f"Built {len(states)} abstract states from {len(raw_screens)} raw screens")
        return sid_to_state_id, states

    @staticmethod
    def _compute_functional_fingerprint(screen: dict[str, Any]) -> str:
        """Compute a functional fingerprint based on page identity.

        Fingerprint priority:
        1. If page_title exists: (title, modal) — title is the primary page identity.
           Container signature is ignored because scrolling changes visible containers.
        2. If no title but container_sig is specific (≥2 classes): (container_sig, modal).
        3. Otherwise: fall back to scroll-aware structural fingerprint.
        """
        metadata = screen.get("metadata", {})
        page_title = metadata.get("page_title", "")
        container_sig = metadata.get("container_signature", "")
        has_modal = metadata.get("has_modal", False)

        if page_title:
            # Title is the primary identity — ignore container (scroll-volatile)
            fp_input = (page_title, has_modal)
            return hashlib.sha256(str(fp_input).encode()).hexdigest()[:16]

        if container_sig:
            # No title — use container sig, but only if specific enough
            num_classes = len(container_sig.split(","))
            if num_classes >= 2:
                fp_input = (container_sig, has_modal)
                return hashlib.sha256(str(fp_input).encode()).hexdigest()[:16]

        # Generic or missing metadata — fall back to structural fingerprint
        return FsmBuilder._compute_structural_fingerprint(screen)

    @staticmethod
    def _compute_structural_fingerprint(screen: dict[str, Any]) -> str:
        """Fallback structural fingerprint excluding scroll-volatile children."""
        elements = screen.get("interactable_elements", screen.get("elements", []))
        if not elements:
            return ""

        scrollable_depths: set[int] = set()
        for e in elements:
            if e.get("is_scrollable"):
                scrollable_depths.add(e.get("depth", 0))

        components = []
        for e in elements:
            depth = e.get("depth", 0)
            if (
                scrollable_depths
                and not e.get("is_scrollable")
                and any(depth > sd for sd in scrollable_depths)
            ):
                continue

            interactability = (
                e.get("is_clickable", False),
                e.get("is_long_clickable", False),
                e.get("is_scrollable", False),
                e.get("is_editable", False),
                e.get("is_checkable", False),
            )
            components.append(
                (
                    e.get("class_name", ""),
                    e.get("resource_id", "") or "",
                    depth,
                    interactability,
                )
            )

        components.sort()
        fingerprint_input = (screen.get("activity_name", "") or "", tuple(components))
        return hashlib.sha256(str(fingerprint_input).encode()).hexdigest()[:16]

    def _build_transitions(
        self,
        raw_traces: list[dict[str, Any]],
        sid_to_state_id: dict[str, str],
        include_self_loops: bool,
        raw_screens: dict[str, Any] | None = None,
    ) -> list[Transition]:
        """Convert exploration traces into FSM transitions.

        Self-loop policy: SCROLL_UP / SCROLL_DOWN / INPUT_TEXT self-loops
        and toggle (is_checkable) clicks are preserved unconditionally,
        because the FSM must represent these as legal affordances even
        when the captured pre/post screens collapsed to the same abstract
        state. Plain CLICK no-op self-loops continue to be dropped unless
        ``include_self_loops`` is set.
        """
        transitions: list[Transition] = []
        skipped_self_loops = 0
        skipped_out_of_app_scope = 0
        partial_scope_observations = 0
        _MEANINGFUL_SELF_LOOPS = {  # noqa: N806
            "scroll_up",
            "scroll_down",
            "input_text",
        }

        for loop_index, trace in enumerate(raw_traces):
            source_sid = trace.get("source_screen_id", "")
            target_sid = trace.get("target_screen_id", "")

            source_state = sid_to_state_id.get(source_sid)
            target_state = sid_to_state_id.get(target_sid)

            if source_state is None or target_state is None:
                continue

            metadata = trace.get("metadata") or {}
            skip_trace, ambiguous_evidence = self._classify_trace_evidence(metadata)
            if skip_trace:
                skipped_out_of_app_scope += 1
                continue
            if ambiguous_evidence:
                partial_scope_observations += 1

            action_data = trace.get("action", {})
            action_type = (action_data.get("action_type") or action_data.get("type") or "").lower()
            is_meaningful_self_loop = action_type in _MEANINGFUL_SELF_LOOPS

            # An observed click self-loop with stable element identity
            # (target_text / resource_id / content-desc) is a verifiable
            # affordance — e.g. Settings list rows like "Airplane mode" or
            # "USB tethering" that stay in the same abstract state after a
            # click. Preserve them. The action dict has not yet been enriched
            # at this point, so we also peek at the source-screen element for
            # identity.
            is_identity_click_self_loop = (
                action_type == "click"
                and source_state == target_state
                and (
                    _click_action_has_identity(action_data)
                    or (raw_screens is not None and self._element_has_identity(trace, raw_screens))
                )
            )

            if (
                not include_self_loops
                and source_state == target_state
                and not is_meaningful_self_loop
                and not is_identity_click_self_loop
                and not (raw_screens and self._is_toggle_action(trace, raw_screens))
            ):
                skipped_self_loops += 1
                continue

            fsm_action, identity_inconsistent = build_trace_action_sample(
                trace,
                raw_screens,
                source_state=source_state,
                target_state=target_state,
            )
            if identity_inconsistent:
                # The action sample did not resolve consistently, so this observation
                # cannot attribute the edge; refinement must not partition on it.
                ambiguous_evidence = True

            raw_action_sample = dict(fsm_action)
            fsm_action = canonical_transition_action(fsm_action)

            step_index = trace.get("step_number")
            if not isinstance(step_index, int):
                step_index = loop_index

            provenance_entries = [
                ProvenanceEntry(
                    trace_step_index=step_index,
                    source_screen_id=source_sid or None,
                    target_screen_id=target_sid or None,
                    raw_action=raw_action_sample,
                    evidence_source="observed",
                )
            ]

            transitions.append(
                Transition(
                    source=source_state,
                    target=target_state,
                    action=fsm_action,
                    ambiguous_evidence=ambiguous_evidence,
                    observed_count=1,
                    provenance=provenance_entries,
                )
            )

        if skipped_self_loops:
            logger.debug(f"Skipped {skipped_self_loops} self-loop transitions")
        if skipped_out_of_app_scope:
            logger.debug(f"Skipped {skipped_out_of_app_scope} out-of-app scope traces")
        if partial_scope_observations:
            logger.debug(
                f"Marked {partial_scope_observations} partial-scope observations as "
                "unattributable for refinement"
            )
        return transitions

    @staticmethod
    def _is_toggle_action(trace: dict[str, Any], raw_screens: dict[str, Any]) -> bool:
        """Check if a trace step targets a checkable element (toggle/switch)."""
        action_data = trace.get("action", {})
        target_eid = action_data.get("target_element_id")
        if not target_eid:
            return False

        source_sid = trace.get("source_screen_id", "")
        screen = raw_screens.get(source_sid, {})
        elements = screen.get("interactable_elements", screen.get("elements", []))

        for el in elements:
            if el.get("element_id") == target_eid:
                return el.get("is_checkable", False)

        return False

    @staticmethod
    def _element_has_identity(trace: dict[str, Any], raw_screens: dict[str, Any]) -> bool:
        """True iff the trace's source-screen element carries stable identity
        (text / resource_id / content-description). Used pre-enrichment so an
        identity-bearing click self-loop is not dropped by ``_build_transitions``
        before the action dict is enriched with element metadata."""
        action_data = trace.get("action", {})
        target_eid = action_data.get("target_element_id")
        if not target_eid:
            return False

        source_sid = trace.get("source_screen_id", "")
        screen = raw_screens.get(source_sid, {})
        elements = screen.get("interactable_elements", screen.get("elements", []))

        for el in elements:
            if el.get("element_id") == target_eid:
                return bool(
                    (el.get("text") or "").strip()
                    or (el.get("content_description") or "").strip()
                    or (el.get("resource_id") or "").strip()
                )
        return False

    def _merge_transitions(self, transitions: list[Transition]) -> list[Transition]:
        """Merge duplicate verifier-equivalent transitions.

        Sums observed_count for duplicates.
        """
        key_to_trans: dict[tuple[Any, ...], Transition] = {}

        for t in transitions:
            t.action = canonical_transition_action(t.action)
            key = transition_identity_key(t)

            if key in key_to_trans:
                self._merge_transition_evidence(key_to_trans[key], t)
            else:
                key_to_trans[key] = t

        merged = list(key_to_trans.values())
        if len(transitions) != len(merged):
            logger.debug(f"Merged {len(transitions)} transitions → {len(merged)} unique")
        return merged

    def _detect_initial_state(
        self,
        raw_traces: list[dict[str, Any]],
        sid_to_state_id: dict[str, str],
        states: dict[str, AbstractState] | None = None,
        app_prior: AppPrior | None = None,
    ) -> str | None:
        """Detect the initial state.

        Preference order:
          1. The state whose ``activity_name`` matches the launcher activity
             declared in ``AppPrior`` (manifest ``MAIN/LAUNCHER`` intent
             filter). Matching tolerates short-class-name equality so the
             launcher ``.MainActivity`` notation in the manifest aligns with
             the fully-qualified runtime ``com.example.app.MainActivity``.
          2. Fallback: the source of the earliest-numbered trace step.
        """
        entry_activity = app_prior.entry_activity if app_prior else None
        if entry_activity and states:
            entry_short = entry_activity.rsplit(".", 1)[-1]
            for state in states.values():
                activity_name = state.android_context.activity_name
                if not activity_name:
                    continue
                if activity_name == entry_activity:
                    return state.state_id
                if activity_name.rsplit(".", 1)[-1] == entry_short:
                    return state.state_id

        if not raw_traces:
            return None
        sorted_traces = sorted(raw_traces, key=lambda t: t.get("step_number", 0))
        first_source = sorted_traces[0].get("source_screen_id", "")
        return sid_to_state_id.get(first_source)

    @staticmethod
    def _disambiguate_names(states: dict[str, AbstractState]) -> None:
        """Append numeric suffixes to duplicate state names."""
        name_counts: dict[str, list[str]] = defaultdict(list)
        for state in states.values():
            name_counts[state.name].append(state.state_id)

        for name, state_ids in name_counts.items():
            if len(state_ids) <= 1:
                continue
            for i, sid in enumerate(state_ids, start=1):
                states[sid].name = f"{name} #{i}"

    def _infer_hierarchy(self, states: dict[str, AbstractState]) -> None:
        """Set hierarchy levels based on activity names.

        States with the same activity_name are FRAGMENT level under
        a shared ACTIVITY parent. States with unique or None activity
        default to ACTIVITY level.
        """
        activity_groups: dict[str | None, list[str]] = defaultdict(list)
        for state in states.values():
            activity_groups[state.android_context.activity_name].append(state.state_id)

        for activity_name, state_ids in activity_groups.items():
            if activity_name is None:
                # No activity info — keep as ACTIVITY level
                continue
            if len(state_ids) > 1:
                # Multiple states share an activity — mark as FRAGMENT
                for sid in state_ids:
                    states[sid].hierarchy_level = HierarchyLevel.FRAGMENT

    def _derive_state_name(
        self,
        screen: dict[str, Any],
        fallback_id: str,
        trace_dir: Path | None = None,
        app_prior: AppPrior | None = None,
    ) -> str:
        """Derive a human-readable state name from screen metadata.

        Priority: Activity label → page_title from XML → first heading → fallback.
        """
        activity_name = screen.get("activity_name") or ""

        # Strategy 1: Manifest Activity label
        if app_prior and activity_name:
            for act in app_prior.activities:
                name_match = act.name == activity_name or (
                    activity_name.rsplit(".", 1)[-1] == act.name.rsplit(".", 1)[-1]
                )
                if name_match and act.label:
                    return act.label

        # Strategy 2: page_title from XML (action_bar_title)
        all_elements = self._get_all_elements(screen, trace_dir)

        for el in all_elements:
            rid = el.get("resource_id", "") or ""
            if "action_bar_title" in rid.lower():
                text = el.get("text")
                if text and text.strip():
                    return text.strip()

        for el in all_elements:
            rid = el.get("resource_id", "") or ""
            if rid and "title" in rid.lower() and "subtitle" not in rid.lower():
                text = el.get("text")
                if text and text.strip() and len(text.strip()) > 1:
                    return text.strip()

        # Strategy 3: Short activity class name (skip the bare framework
        # base-class name, which carries no page identity).
        if activity_name:
            short = activity_name.rsplit(".", 1)[-1]
            if short and short != "Activity":
                return short

        # Strategy 4: first non-empty text from interactable elements
        interactable = screen.get("interactable_elements", screen.get("elements", []))
        for el in interactable:
            text = el.get("text")
            if text and text.strip() and len(text.strip()) > 2:
                return text.strip()

        return fallback_id

    def _get_all_elements(
        self, screen: dict[str, Any], trace_dir: Path | None = None
    ) -> list[dict[str, Any]]:
        """Get all elements for a screen, parsing XML if available.

        Falls back to interactable_elements if XML is not found.
        """
        xml_rel_path = screen.get("xml_tree_path")
        if xml_rel_path and trace_dir is not None:
            xml_path = self._resolve_path(xml_rel_path, trace_dir)
            if xml_path is not None:
                xml_content = xml_path.read_text(encoding="utf-8")
                elements = parse_hierarchy_xml(xml_content)
                if elements:
                    return [e.model_dump() for e in elements]

        return screen.get("interactable_elements", screen.get("elements", []))

    @staticmethod
    def _resolve_path(rel_path: str, trace_dir: Path) -> Path | None:
        """Resolve a path, trying multiple strategies.

        Order: absolute → CWD-relative → trace_dir-relative → trees/ sibling.
        """
        p = Path(rel_path)
        # 1. Absolute path
        if p.is_absolute() and p.exists():
            return p
        # 2. Relative to CWD (covers project-root-relative paths like
        #    "data/apps/settings/trees/scr_0001.xml")
        if p.exists():
            return p
        # 3. Relative to trace dir
        candidate = trace_dir / rel_path
        if candidate.exists():
            return candidate
        # 4. Try resolving just the filename in the trees/ sibling directory
        trees_dir = trace_dir.parent / "trees"
        if trees_dir.is_dir():
            candidate = trees_dir / p.name
            if candidate.exists():
                return candidate
        return None


def assign_member_fingerprints(fsm: AppFSM, raw_screens: dict[str, dict[str, Any]]) -> int:
    """Populate ``StateIdentity.member_fingerprints`` for every state in ``fsm``.

    A state's members are the live-screen identities of the raw screens it was built
    from -- the same key ``StateLocator`` computes for a live screen at runtime. The
    identity of a trace screen is its explorer-computed ``state_id`` when present and
    the scroll-aware functional fingerprint otherwise, exactly as
    :meth:`FsmBuilder._build_states_from_screens` derives it, so a member entry and a
    runtime lookup key are produced by the same rule.

    Quotienting absorbs many identities into one state but keeps only the representative
    in ``functional_hash``; this pass restores the full set from
    ``evidence.raw_screen_ids`` after every merge, split and redirect has run, so it
    cannot disagree with the partition the builder actually produced. Screens the
    caller no longer has (ids present in evidence but absent from ``raw_screens``) are
    skipped rather than guessed.

    ``scripts/migrate_fsm_v5_to_v6.py`` calls this same function to backfill FSMs
    serialized before schema v6, so the builder and the migration can never diverge.

    Args:
        fsm: The FSM to annotate, mutated in place.
        raw_screens: ``screen_id -> screen dict`` from the exploration trace.

    Returns:
        Total number of member identities assigned across all states.
    """
    total = 0
    for state in fsm.states.values():
        members = member_fingerprints_for(
            state.evidence.raw_screen_ids,
            raw_screens,
            representative=state.identity.functional_hash,
        )
        state.identity.member_fingerprints = members
        total += len(members)
    return total


def member_fingerprints_for(
    raw_screen_ids: list[str],
    raw_screens: dict[str, dict[str, Any]],
    *,
    representative: str | None = None,
) -> list[str]:
    """Sorted, deduplicated live-screen identities for one state's raw screens.

    Split out of :func:`assign_member_fingerprints` so the on-disk backfill in
    ``scripts/migrate_fsm_v5_to_v6.py`` can reuse the identical rule while working on a
    raw JSON payload -- neither path can drift from the other, and the migration never
    has to round-trip a stored bundle through the model.

    ``representative`` (the state's ``functional_hash``) is always included when
    non-empty, so a state whose screens were pruned from the trace still indexes itself.
    Screen ids absent from ``raw_screens`` are skipped, never guessed.
    """
    members: set[str] = set()
    for screen_id in raw_screen_ids:
        screen = raw_screens.get(screen_id)
        if not screen:
            continue
        identity = screen.get("state_id") or FsmBuilder._compute_functional_fingerprint(screen)
        if identity:
            members.add(identity)
    if representative:
        members.add(representative)
    return sorted(members)

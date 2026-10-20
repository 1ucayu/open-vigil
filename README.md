# Vigil

Vigil automates symbolic verification for mobile GUI agents. It constructs an
extended finite-state machine (EFSM) from recorded Android observations, admits
neural guardrail candidates through deterministic checks, and uses the resulting
model for plan-time guidance, an execution-time shield, and evidence-only feedback.

This repository contains the system implementation, packaged DSL and prompts,
configuration, and command-line interfaces.

## Installation

Use Python 3.11 or newer and `uv`:

```sh
uv sync
```

Live exploration also requires Android platform-tools (`adb`) on `PATH` and an
authorized Android device or emulator. Inspect available devices with `adb devices`.
Model construction from existing traces and symbolic verification do not require a
connected device.

## Construct an EFSM

Collect observations from an app:

```sh
uv run vigil-explore --app com.example.app --serial DEVICE_SERIAL --steps 100 \
  --output-dir artifacts/app/observations --prior-from-device
```

Use the generated trace JSON to build the structural model:

```sh
uv run vigil-build --trace /path/to/exploration.json \
  --output artifacts/app/explored_fsm/fsm.json
```

Neural construction uses a configurable API endpoint. Set the model and connection
details in the environment, or provide a YAML file with `--config`:

```sh
export VIGIL_LLM_BASE_URL="https://your-endpoint.example/v1"
export VIGIL_LLM_MODEL="your-model-id"
export VIGIL_LLM_API_KEY="your-api-key"
```

`VIGIL_LLM_MODELS_URL` optionally overrides the model metadata endpoint, which
otherwise defaults to `VIGIL_LLM_BASE_URL/models`. `OPENAI_BASE_URL` and
`OPENAI_API_KEY` are accepted as fallback environment settings. Keep credentials
in the environment or an ignored `configs/local.yaml`.

Enrich the structural model with visual annotations and admitted guardrails:

```sh
uv run vigil-build --trace /path/to/exploration.json \
  --fsm artifacts/app/explored_fsm/fsm.json \
  --visual-ground --generate-guards --generate-invariants \
  --output artifacts/app/enriched_efsm/efsm.json
```

Guardrail admission preserves the states, canonical actions, and observed edges.
Transition contracts are attached atomically for each source state after individual
admission and shared intent-slot schema checks. The input FSM remains unchanged;
the enriched model and admission diagnostics are written beside each other. The
`artifacts/` directory is ignored by Git.

## Verify proposed actions

The CLI checks an action against a captured accessibility hierarchy without
executing it:

```sh
uv run vigil-verify-action --fsm artifacts/app/enriched_efsm/efsm.json \
  --screen /path/to/screen.xml --runtime \
  --activity com.example.app.MainActivity --package com.example.app \
  --action '{"type":"click","target":"e_0042"}' \
  --bindings '{"max_order_total":30}'
```

Use an element identifier and the foreground activity/package captured with the
supplied observation. Activity context participates in state identity; omitting it
can leave localization unresolved. Intent bindings are
frozen values supplied by the agent. Only `ALLOW` permits execution; `DENY`
reports a concretely refuted condition, and `UNCERTAIN` reports missing or
inconclusive evidence. The symbolic shield does not invoke an LLM.

For an agent integration, `RuntimeMonitor.verify_action()` performs the pre-action
check, and `verify_post_action()` checks the expected successor and its invariants.
The reusable JSON adapter is available through:

```sh
uv run python -m vigil.integration.m3a_adapter --serve
```

It accepts one JSON request per line and emits one JSON response per line. The
adapter implements planning, intent admission, execution checking, post-action
checking, and explicit evidence/evolution requests. Refusals block execution by
default. Additional evidence collection requires a separately authorized policy;
it does not turn an unverified action into an `ALLOW` verdict.

Evolution primitives support observed-transition admission and targeted contract
enrichment. The embedding agent controls collection and activation timing; these
primitives do not themselves orchestrate the paper's complete offline rebuild
between verification rounds. The default explorer follows the paper's breadth-first
strategy; emulator snapshot depth-first exploration is an optional alternative.

## Paper terminology and code

Code uses descriptive English names for the paper's mathematical objects:

| Paper object | Implementation |
| --- | --- |
| Raw observation `o_t` | `RawScreen`, `screen`, `current_screen` |
| Proposed raw action `a_t` | `Action`, `ProposedAction`, `proposed_action` |
| Abstract state `s_t` | `AbstractState`, `state_id`, `source_state_id` |
| Successor state `s_{t+1}` | `target_state_id`, `expected_target_state_id` |
| State set `S` and initial state `s_0` | `AppFSM.states`, `AppFSM.initial_state` |
| Canonical action `sigma_t` in `Sigma` | `canonical_transition_action()`, `canonical_action_key()`, `Transition.action` |
| Transition guardrail `Gamma(s_t, sigma_t)` | `Transition.guard_contract`, `ClauseActivator`, `DSLEvaluator` |
| State invariant `I(s_t)` | `AbstractState.invariant_specs`, `InvariantChecker` |
| Per-app model `M_A` | `AppFSM` |

Canonical action dictionaries use `type` for the action type. Stable GUI identity
uses resource identifiers, text, or content descriptions where available, with
`target_selector` retained when those fields are insufficient. Input-text operations
are parameterized: their
concrete input remains in the proposed action and trace provenance, and guardrails
check that value against frozen intent. It is not part of the structural action
key. `Transition` stores a source, a canonical action, and a successor; transition
storage is explicit even though the paper's model tuple leaves it implicit.

## Source layout

| Package | Responsibility |
| --- | --- |
| `vigil.models` | Observations, actions, EFSMs, guard contracts, intent bindings, and evolution records |
| `vigil.core` | Accessibility parsing, selectors, configuration, device selection, and neural client |
| `vigil.neuro` | App priors, exploration, state abstraction/refinement, guardrail synthesis and admission |
| `vigil.symbolic` | Localization, transition checking, planning guidance, clause activation, and DSL evaluation |
| `vigil.runtime` | Action binding, execution-time verification, successor checking, and evidence collection |
| `vigil.integration` | Reusable agent I/O adapter and runtime evidence transport |
| `vigil.scripts` | Exploration, model construction, and action-verification CLIs |
| `vigil.system_prompt` | Packaged transition and invariant synthesis prompts |

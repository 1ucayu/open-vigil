"""Generic, agent-agnostic runtime verification layer for Vigil.

Wires the existing offline-built FSM and online symbolic verifier pieces
(state localization, action binding, intent binding, decision engine,
invariant checking) into a single deterministic online verification loop:

    screen -> RawScreen -> state localization -> action binding
      -> intent binding -> pre-action VERIFY -> optional execution
      -> post-action localization -> target / invariant check

This package is NOT an adapter for any specific mobile-agent framework. It is
the reusable substrate an adapter (M3A, Mobile-Agent-v2, MobiAgent, ...) sits
on. The common path is symbolic and deterministic: it never routes runtime
decisions through an LLM, never silently ALLOWs on incomplete evidence, and
never weakens canonical action identity.
"""

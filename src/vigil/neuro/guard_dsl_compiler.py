"""Deterministic ``GuardContract`` / ``PredicateSpec`` → DSL string compiler (step 4).

This is a *pure structural renderer*: it turns typed predicates into the current DSL
grammar (``src/vigil/symbolic/dsl_grammar.lark``) with no knowledge of the widget registry or
the runtime context. Element and action-property references are emitted verbatim;
lowering them to runtime-resolvable keys and deciding executability is the job of
:mod:`vigil.neuro.guard_admission`.

No LLM, no grammar expansion. Output uses the canonical grammar's spaced operators,
JSON-quoted string literals, and ``$intent.x`` variables.
"""

from __future__ import annotations

import json
import math

from vigil.models.guard import PredicateSpec, ValueRef

_DEFAULT_OP = "=="
# Mirrors the LLM-facing ``OperatorLiteral``; ``not_contains`` stays runtime-DSL-only.
_VALID_OPS: frozenset[str] = frozenset({"==", "!=", ">", "<", ">=", "<=", "contains"})


def _render_value(ref: ValueRef | None) -> str | None:
    """Render the right-hand VALUE token, or ``None`` if it cannot be expressed."""
    if ref is None:
        return None
    if ref.kind == "literal":
        value = ref.value
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value) if value >= 0 else None
        if isinstance(value, float):
            if value < 0 or not math.isfinite(value):
                return None
            return str(value)
        return json.dumps(str(value))
    return f"$intent.{ref.slot}" if ref.slot else None


def compile_predicate_spec(predicate: PredicateSpec) -> str | None:
    """Compile one :class:`PredicateSpec` to a DSL predicate, or ``None``.

    Returns ``None`` when a required part is missing or the predicate cannot be expressed
    in the current grammar (e.g. an ``action``/``read`` value on the right-hand side).
    """
    ptype = predicate.predicate_type

    if ptype == "read":
        if not predicate.element or not predicate.property:
            return None
        operand = f"read({predicate.element}, {predicate.property})"
    elif ptype in {"value", "count"}:
        if not predicate.element:
            return None
        operand = f"{ptype}({predicate.element})"
    else:
        if not predicate.property:
            return None
        operand = f"action({predicate.property})"

    op = predicate.operator or _DEFAULT_OP
    value = _render_value(predicate.expected)
    if op not in _VALID_OPS or value is None:
        return None
    return f"{operand} {op} {value}"

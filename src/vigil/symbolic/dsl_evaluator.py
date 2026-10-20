"""Deterministic evaluation of admitted DSL guardrails.

Evaluates DSL guard expressions against the current screen state using the Lark
parser shipped with the symbolic verifier. Guard templates are cached offline;
parameters are bound at runtime from user intent.

Evaluation is three-valued (TRUE / FALSE / UNKNOWN) to match the paper model:
proven-false predicates are distinguishable from those the verifier cannot
read (missing element, unbound $intent variable, parse failure, type-coercion
failure, etc.). UNKNOWN routes to UNCERTAIN at the DecisionEngine layer rather
than collapsing to FALSE.
"""

from __future__ import annotations

import ast
import operator
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from lark import Lark, Token, Transformer
from lark.exceptions import UnexpectedInput
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from vigil.core.paths import resolve_dsl_grammar_path
from vigil.symbolic.numeric_normalizer import (
    NumericNormalizationError,
    normalize_numeric_scalar,
)

JsonScalar = str | int | float | bool | None


class IntentContext(BaseModel):
    """Variables extracted from user instruction for guard binding.

    Attributes:
        variables: Named values from the instruction (e.g., {"wifi_name": "HKU_WiFi"}).
    """

    variables: dict[str, JsonScalar] = Field(default_factory=dict)


class ScreenContext(BaseModel):
    """Runtime screen state for predicate evaluation.

    Attributes:
        elements: element_id → property dict (e.g., {"text": "HKU", "checked": True}).
    """

    elements: dict[str, dict[str, Any]] = Field(default_factory=dict)


# Sentinel property set by the ScreenContext builder when one resource_id alias
# is claimed by several distinct live elements; predicates over that alias must
# evaluate UNKNOWN rather than read an arbitrary element.
AMBIGUOUS_RESOURCE_ID_MARKER = "_ambiguous_resource_id"


class GuardStatus(StrEnum):
    """Three-valued evaluation result for a DSL guard or invariant."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class PredicateWitness(BaseModel):
    """One predicate's operands and result from the real evaluator pass.

    This is captured *during* evaluation, so it reflects exactly the surfaces and
    operands the evaluator compared — it is not a separate re-parse or re-read.
    ``element``/``property`` describe the screen/action operand; ``expected`` is
    the resolved right-hand value (``$intent.*`` already substituted); ``actual``
    is the scalar surface read (``None`` when the operand was a container/list or
    unreadable). Predicates appear in evaluation order, one per admitted clause.
    """

    model_config = ConfigDict(frozen=True)

    predicate_type: str
    element: str | None = None
    property: str | None = None
    operator: str = ""
    expected: JsonScalar = None
    actual: JsonScalar = None
    result: GuardStatus
    reason: str = ""


def _json_scalar(value: Any) -> JsonScalar:
    """Coerce an operand to a JSON scalar for witness reporting (else ``None``)."""
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return None


class GuardResult(BaseModel):
    """Result of evaluating a DSL guard.

    Attributes:
        status: TRUE, FALSE, or UNKNOWN.
        guard_expression: The original guard expression.
        failure_reason: Why the guard is false or unknown (empty if true).
        witness: Per-predicate operands/results from this pass (when requested).
    """

    status: GuardStatus = GuardStatus.FALSE
    guard_expression: str
    failure_reason: str = ""
    witness: tuple[PredicateWitness, ...] | None = None


# Operator lookup
_OPS: dict[str, Any] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}
_CONTAINMENT_OPS = frozenset({"contains", "not_contains"})


class _Unknown:
    """Sentinel value representing an inconclusive predicate result."""

    __slots__ = ("reason",)

    def __init__(self, reason: str = "") -> None:
        self.reason = reason

    def __bool__(self) -> bool:  # pragma: no cover — guarded by callers
        return False

    def __repr__(self) -> str:
        return f"<UNKNOWN: {self.reason}>"


def _is_unknown(v: Any) -> bool:
    return isinstance(v, _Unknown)


def _normalize_text_for_compare(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


# Public alias. Guard admission compares recorded source readings to decide whether a
# conjunction is jointly satisfiable, and that comparison must be the *same* one the
# shield will perform at runtime — never a second, drifting notion of text equality.
normalize_text_for_compare = _normalize_text_for_compare


_STRUCTURAL_TOKENS = frozenset(
    {
        "read(",
        "value(",
        "count(",
        "action(",
        ",",
        ")",
        "(",
        "&&",
    }
)


def _filter_named(args: list[Any]) -> list[Any]:
    """Filter out anonymous structural tokens, keeping only named tokens and results."""
    return [a for a in args if not (isinstance(a, Token) and str(a) in _STRUCTURAL_TOKENS)]


class _GuardEvaluator(Transformer):
    """Lark Transformer that evaluates a parsed guard tree against screen context.

    Each predicate handler returns either True, False, or `_Unknown(reason)`.
    The conjunction rule implements three-valued logic:
    ``UNKNOWN && False -> False`` and ``UNKNOWN && True -> UNKNOWN``.
    """

    def __init__(
        self,
        screen_ctx: ScreenContext,
        intent_ctx: IntentContext | None = None,
        action_ctx: dict[str, Any] | None = None,
        witness: list[PredicateWitness] | None = None,
    ) -> None:
        super().__init__()
        self._ctx = screen_ctx
        self._intent = intent_ctx
        self._action = action_ctx
        self._witness = witness

    def _record(
        self,
        predicate_type: str,
        operator: str,
        result: Any,
        *,
        element: str | None = None,
        prop: str | None = None,
        expected: Any = None,
        actual: Any = None,
    ) -> Any:
        """Append one predicate witness (when recording) and return ``result``.

        Recording is passive: it never changes ``result`` and never re-runs a
        comparison. ``_compare``/``_contains`` semantics are untouched.
        """
        if self._witness is not None:
            if _is_unknown(result):
                status, reason = GuardStatus.UNKNOWN, result.reason
            elif result is True:
                status, reason = GuardStatus.TRUE, ""
            else:
                status, reason = GuardStatus.FALSE, ""
            self._witness.append(
                PredicateWitness(
                    predicate_type=predicate_type,
                    element=element,
                    property=prop,
                    operator=operator,
                    expected=(None if _is_unknown(expected) else _json_scalar(expected)),
                    actual=(None if _is_unknown(actual) else _json_scalar(actual)),
                    result=status,
                    reason=reason,
                )
            )
        return result

    def start(self, args: list[Any]) -> Any:
        return args[0]

    def guard(self, args: list[Any]) -> Any:
        token_strs = [str(a) for a in args if isinstance(a, Token)]
        filtered = _filter_named(args)

        if "&&" in token_strs:
            a, b = filtered[0], filtered[1]
            # Three-valued AND
            if a is False or b is False:
                return False
            if _is_unknown(a) or _is_unknown(b):
                return _Unknown("conjunction has UNKNOWN operand")
            return bool(a) and bool(b)

        # Single predicate.
        return filtered[0]

    def predicate(self, args: list[Any]) -> Any:
        return args[0]

    def read_pred(self, args: list[Any]) -> Any:
        named = _filter_named(args)
        element_name = str(named[0])
        prop_name = str(named[1])
        op_str = str(named[2])
        expected = self._parse_value(named[3])
        if _is_unknown(expected):
            return self._record("read", op_str, expected, element=element_name, prop=prop_name)

        el = self._ctx.elements.get(element_name)
        if el is None:
            return self._record(
                "read",
                op_str,
                _Unknown(f"element '{element_name}' not present on screen"),
                element=element_name,
                prop=prop_name,
                expected=expected,
            )
        if el.get(AMBIGUOUS_RESOURCE_ID_MARKER):
            return self._record(
                "read",
                op_str,
                _Unknown(f"resource_id not unique on live screen: '{element_name}'"),
                element=element_name,
                prop=prop_name,
                expected=expected,
            )
        if prop_name not in el:
            return self._record(
                "read",
                op_str,
                _Unknown(f"property '{prop_name}' not readable on '{element_name}'"),
                element=element_name,
                prop=prop_name,
                expected=expected,
            )
        actual = el.get(prop_name)
        return self._record(
            "read",
            op_str,
            self._compare(actual, op_str, expected),
            element=element_name,
            prop=prop_name,
            expected=expected,
            actual=actual,
        )

    def value_pred(self, args: list[Any]) -> Any:
        named = _filter_named(args)
        element_name = str(named[0])
        op_str = str(named[1])
        expected = self._parse_value(named[2])
        if _is_unknown(expected):
            return self._record("value", op_str, expected, element=element_name)

        el = self._ctx.elements.get(element_name)
        if el is None:
            return self._record(
                "value",
                op_str,
                _Unknown(f"element '{element_name}' not present on screen"),
                element=element_name,
                expected=expected,
            )
        if el.get(AMBIGUOUS_RESOURCE_ID_MARKER):
            return self._record(
                "value",
                op_str,
                _Unknown(f"resource_id not unique on live screen: '{element_name}'"),
                element=element_name,
                expected=expected,
            )
        if "value" not in el and "text" not in el and op_str not in _CONTAINMENT_OPS:
            return self._record(
                "value",
                op_str,
                _Unknown(f"no value on '{element_name}'"),
                element=element_name,
                expected=expected,
            )
        # Containment over ``value(container)`` is defined over the container's
        # visible child values. Runtime ScreenContext also carries empty scalar
        # ``value``/``text`` keys for containers, so pass the whole element record
        # instead of letting those placeholders hide its children.
        actual = el if op_str in _CONTAINMENT_OPS else el.get("value", el.get("text", el))
        return self._record(
            "value",
            op_str,
            self._compare(actual, op_str, expected),
            element=element_name,
            expected=expected,
            actual=actual,
        )

    def count_pred(self, args: list[Any]) -> Any:
        named = _filter_named(args)
        element_name = str(named[0])
        op_str = str(named[1])
        expected = self._parse_value(named[2])
        if _is_unknown(expected):
            return self._record("count", op_str, expected, element=element_name)

        el = self._ctx.elements.get(element_name)
        if el is None:
            return self._record(
                "count",
                op_str,
                _Unknown(f"element '{element_name}' not present on screen"),
                element=element_name,
                expected=expected,
            )
        if el.get(AMBIGUOUS_RESOURCE_ID_MARKER):
            return self._record(
                "count",
                op_str,
                _Unknown(f"resource_id not unique on live screen: '{element_name}'"),
                element=element_name,
                expected=expected,
            )
        if "children_count" not in el:
            return self._record(
                "count",
                op_str,
                _Unknown(f"no count on '{element_name}'"),
                element=element_name,
                expected=expected,
            )
        count = el["children_count"]
        return self._record(
            "count",
            op_str,
            self._compare(count, op_str, expected),
            element=element_name,
            expected=expected,
            actual=count,
        )

    def action_pred(self, args: list[Any]) -> Any:
        named = _filter_named(args)
        prop_name = str(named[0])
        op_str = str(named[1])
        expected = self._parse_value(named[2])
        if _is_unknown(expected):
            return self._record("action", op_str, expected, prop=prop_name)

        if self._action is None:
            return self._record(
                "action",
                op_str,
                _Unknown("action_ctx not provided"),
                prop=prop_name,
                expected=expected,
            )
        if prop_name not in self._action:
            return self._record(
                "action",
                op_str,
                _Unknown(f"action property '{prop_name}' missing"),
                prop=prop_name,
                expected=expected,
            )
        actual = self._action.get(prop_name)
        return self._record(
            "action",
            op_str,
            self._compare(actual, op_str, expected, source_numeric=False),
            prop=prop_name,
            expected=expected,
            actual=actual,
        )

    def _parse_value(self, token: Token | str) -> Any:
        """Parse a VALUE token into a Python value, resolving $intent.* from context.

        Returns `_Unknown(reason)` if an $intent variable is referenced but
        absent from the IntentContext (or no IntentContext was supplied).
        """
        s = str(token)
        if s.startswith("$intent."):
            var_name = s[len("$intent.") :]
            if self._intent is None or var_name not in self._intent.variables:
                return _Unknown(f"$intent.{var_name} not bound")
            return self._intent.variables[var_name]
        if s == "true":
            return True
        if s == "false":
            return False
        if s.startswith('"') and s.endswith('"'):
            try:
                return ast.literal_eval(s)
            except (SyntaxError, ValueError):
                return s[1:-1]
        try:
            if any(marker in s for marker in (".", "e", "E")):
                return Decimal(s)
            return int(s)
        except (ValueError, InvalidOperation):
            return _Unknown("numeric literal exceeds supported limits")

    @staticmethod
    def _contains(actual: Any, expected: Any) -> Any:
        """Evaluate containment over strings, lists, or runtime element dictionaries.

        Runtime element dictionaries expose descendant text through ``children``.
        Each descendant is an independent visible text surface and therefore uses
        substring semantics. Bare lists retain exact-membership semantics.
        """
        normalized_expected = _normalize_text_for_compare(expected)
        if isinstance(normalized_expected, str) and not normalized_expected:
            return _Unknown("text containment requires a non-empty expected value")
        if actual is None:
            return _Unknown("actual value is None")
        if isinstance(actual, dict):
            has_readable_surface = False
            children = actual.get("children")
            if isinstance(children, list):
                has_readable_surface = True
                for child in children:
                    if isinstance(child, dict):
                        child_text = child.get("text")
                        if child_text is not None and str(normalized_expected) in str(
                            _normalize_text_for_compare(child_text)
                        ):
                            return True
                    elif _normalize_text_for_compare(child) == normalized_expected:
                        return True
            for key in ("value", "text"):
                if key not in actual or actual[key] is None:
                    continue
                has_readable_surface = True
                if str(normalized_expected) in str(_normalize_text_for_compare(actual[key])):
                    return True
            if has_readable_surface:
                return False
            return _Unknown("no text/value/children for containment")
        if isinstance(actual, list):
            for item in actual:
                if (
                    isinstance(item, dict)
                    and _normalize_text_for_compare(item.get("text")) == normalized_expected
                ):
                    return True
                if _normalize_text_for_compare(item) == normalized_expected:
                    return True
            return False
        return str(normalized_expected) in str(_normalize_text_for_compare(actual))

    @classmethod
    def _compare(
        cls,
        actual: Any,
        op_str: str,
        expected: Any,
        *,
        source_numeric: bool = True,
    ) -> Any:
        """Three-valued comparison with type coercion for numeric comparisons."""
        if isinstance(expected, int | float | Decimal) and not isinstance(expected, bool):
            if op_str in _CONTAINMENT_OPS:
                return _Unknown("numeric intent values cannot use text containment")
            op_func = _OPS.get(op_str)
            if op_func is None:
                return _Unknown(f"unsupported operator '{op_str}'")
            try:
                actual_number = normalize_numeric_scalar(
                    actual,
                    source_value=source_numeric,
                )
                expected_number = normalize_numeric_scalar(
                    expected,
                    source_value=False,
                )
            except NumericNormalizationError as exc:
                return _Unknown(str(exc))
            return op_func(actual_number, expected_number)
        if op_str in _CONTAINMENT_OPS:
            result = cls._contains(actual, expected)
            if _is_unknown(result):
                return result
            return (not result) if op_str == "not_contains" else result
        op_func = _OPS.get(op_str)
        if op_func is None:
            return _Unknown(f"unsupported operator '{op_str}'")
        if actual is None:
            return _Unknown("actual value is None")
        if isinstance(actual, str) and isinstance(expected, str):
            actual = _normalize_text_for_compare(actual)
            expected = _normalize_text_for_compare(expected)
        try:
            return op_func(actual, expected)
        except TypeError:
            return _Unknown("type mismatch in comparison")


class DSLEvaluator:
    """Evaluates DSL guard expressions against runtime context.

    Uses the packaged Lark grammar for parsing.
    Guard templates contain $intent.* placeholders resolved at runtime via
    INTENT_VAR tokens in the grammar.
    """

    def __init__(self) -> None:
        grammar_text = resolve_dsl_grammar_path().read_text(encoding="utf-8")
        self._parser = Lark(grammar_text, parser="earley", start="start", keep_all_tokens=True)

    def evaluate(
        self,
        guard_expr: str,
        intent_ctx: IntentContext | None = None,
        screen_ctx: ScreenContext | None = None,
        action_ctx: dict[str, Any] | None = None,
        record_witness: bool = False,
    ) -> GuardResult:
        """Parse guard, resolve $intent.* variables, evaluate predicates.

        When ``record_witness`` is set, the returned :class:`GuardResult` carries a
        per-predicate ``witness`` capturing the operands and result of this exact
        pass (never a re-parse or re-read).
        """
        if screen_ctx is None:
            screen_ctx = ScreenContext()

        witness: list[PredicateWitness] | None = [] if record_witness else None

        def frozen_witness() -> tuple[PredicateWitness, ...] | None:
            return None if witness is None else tuple(witness)

        try:
            tree = self._parser.parse(guard_expr)
        except UnexpectedInput as e:
            logger.debug(f"Guard parse error: {e}")
            return GuardResult(
                status=GuardStatus.UNKNOWN,
                guard_expression=guard_expr,
                failure_reason=f"Parse error: {e}",
                witness=frozen_witness(),
            )

        evaluator = _GuardEvaluator(
            screen_ctx,
            intent_ctx=intent_ctx,
            action_ctx=action_ctx,
            witness=witness,
        )
        result = evaluator.transform(tree)

        if _is_unknown(result):
            return GuardResult(
                status=GuardStatus.UNKNOWN,
                guard_expression=guard_expr,
                failure_reason=f"Inconclusive: {result.reason}",
                witness=frozen_witness(),
            )
        if result is True:
            return GuardResult(
                status=GuardStatus.TRUE,
                guard_expression=guard_expr,
                failure_reason="",
                witness=frozen_witness(),
            )
        return GuardResult(
            status=GuardStatus.FALSE,
            guard_expression=guard_expr,
            failure_reason="Guard condition not satisfied",
            witness=frozen_witness(),
        )

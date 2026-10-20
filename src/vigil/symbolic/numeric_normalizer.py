"""Deterministic scalar normalization for numeric guard operands.

Vigil has one numeric type.  The mobile agent binds a number and the verifier compares
it with a source value containing exactly one scalar.  This module does not model units,
currencies, dimensions, locales, or conversions.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_SCALAR_PATTERN = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


class NumericNormalizationError(ValueError):
    """The supplied value does not expose one unambiguous numeric scalar."""


def _single_scalar_match(rendered: str) -> re.Match[str]:
    matches = list(_SCALAR_PATTERN.finditer(rendered))
    if len(matches) != 1:
        raise NumericNormalizationError(
            f"expected exactly one numeric scalar in source value: {rendered!r}"
        )
    match = matches[0]
    preceding = rendered[match.start() - 1] if match.start() else ""
    following = rendered[match.end()] if match.end() < len(rendered) else ""
    if preceding.isalnum() or preceding == "_" or following.isalnum() or following == "_":
        raise NumericNormalizationError(
            f"numeric scalar is embedded in an identifier: {rendered!r}"
        )
    return match


def normalize_numeric_scalar(value: object, *, source_value: bool) -> Decimal:
    """Return a finite ``Decimal`` for an intent or UI value.

    Intent values are canonical scalars.  Source strings may include stable display text
    such as ``"Total: $4.50"``, but must contain exactly one standalone scalar.  Multiple
    scalars remain ambiguous and therefore unevaluable.
    """
    if value is None or isinstance(value, bool):
        raise NumericNormalizationError("numeric value must be an integer or decimal")

    if isinstance(value, int | float | Decimal):
        token = str(value)
    elif isinstance(value, str):
        rendered = value.strip()
        token = _single_scalar_match(rendered).group(0) if source_value else rendered
    else:
        raise NumericNormalizationError("unsupported numeric value type")

    if _SCALAR_PATTERN.fullmatch(token) is None:
        raise NumericNormalizationError(f"not a canonical numeric scalar: {token!r}")

    try:
        return Decimal(token)
    except InvalidOperation as exc:
        raise NumericNormalizationError(f"invalid decimal scalar: {token!r}") from exc

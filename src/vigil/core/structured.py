"""Provider-neutral structured-output result type and JSON-Schema helpers.

These support the schema-constrained guard/invariant generation path:

    Pydantic strict response model -> provider structured output -> parsed typed object

:class:`StructuredResult` is what :meth:`LlmClient.generate_structured` returns. It records
the parsed Pydantic object (when available), the raw provider text/payload summary, the
provider/model, the schema name, the *constraint mode* actually used by the provider,
and any refusal / incomplete / validation metadata. There are no token-budget fields here:
the project assumes an unlimited output budget and never truncates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

# How strongly the provider constrained the output shape:
# - native_schema: provider enforced the JSON Schema (openai json_schema / google response_schema)
# - native_schema_unenforced: provider accepted the native request but returned invalid content
# - tool_schema: provider forced a tool whose input_schema is the JSON Schema (anthropic)
# - prompt_only_unavailable: no structured-output path available
SchemaConstraintMode = Literal[
    "native_schema",
    "native_schema_unenforced",
    "tool_schema",
    "prompt_only_unavailable",
]


@dataclass
class StructuredResult:
    """Outcome of a structured-generation request.

    ``parsed`` is the validated ``response_model`` instance, or ``None`` when the provider
    could not produce a schema-valid object (refused, returned an unsupported-schema error,
    or emitted output that failed validation). Callers MUST treat ``parsed is None`` as a
    clear failure: never fabricate an empty object and never run downstream admission as if
    generation had succeeded.
    """

    parsed: BaseModel | None
    raw_text: str
    provider: str
    model: str
    schema_name: str
    schema_constraint_mode: SchemaConstraintMode
    refusal: str | None = None
    incomplete: bool = False
    incomplete_detail: str | None = None
    validation_errors: list[str] = field(default_factory=list)


def to_strict_schema(model: type[BaseModel], *, inline_refs: bool = False) -> dict[str, Any]:
    """Return ``model``'s JSON Schema tightened for strict structured output.

    Every object node gets ``additionalProperties: false`` and a ``required`` list covering
    all of its declared properties. A bare top-level ``anyOf`` (a union root model) is
    rejected because no supported provider accepts it as a strict response schema. When
    ``inline_refs`` is set, ``$ref``/``$defs`` are inlined for providers that cannot resolve
    local references.
    """
    schema = model.model_json_schema()
    if "anyOf" in schema and "properties" not in schema:
        raise ValueError(f"{model.__name__}: root schema must be an object, not a top-level anyOf")
    if inline_refs:
        schema = _inline_refs(schema)
    _strictify(schema)
    return schema


def _strictify(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            props = node.get("properties")
            if isinstance(props, dict):
                node["required"] = list(props.keys())
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for item in node:
            _strictify(item)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                return resolve(defs[ref.split("/")[-1]])
            return {key: resolve(value) for key, value in node.items() if key != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)

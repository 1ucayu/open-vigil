"""Neural proposal client for construction and between-round contract enrichment.

Supports OpenAI-compatible text and vision endpoints. Neural output is always a
candidate for deterministic admission; the execution-time shield never calls it.
"""

from __future__ import annotations

import base64
import io
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from PIL import Image
from pydantic import BaseModel, ValidationError

from vigil.core.config import LLMConfig
from vigil.core.structured import (
    SchemaConstraintMode,
    StructuredResult,
    to_strict_schema,
)

_MAX_IMAGE_EDGE = 1280
_JPEG_QUALITY = 85
_MAX_RETRIES = 3
_BACKOFF_BASE = 1  # seconds

# Anthropic's Messages API REQUIRES max_tokens (no default; omitting it raises). This is the
# only output cap in this module and exists purely to satisfy the API contract, not to limit
# generation. It is set to a large model-output ceiling, never a cost/token budget.
_ANTHROPIC_NON_STREAMING_MAX_OUTPUT_TOKENS = 16000
_API_POST_READ_TIMEOUT_SECONDS = 90

# Substrings that mark a provider/server rejecting the JSON-Schema structured-output request
# (as opposed to an unrelated 400). Matched case-insensitively against the error text.
_SCHEMA_UNSUPPORTED_MARKERS = (
    "json_schema",
    "response_format",
    "response_schema",
    "additionalproperties",
    "not supported",
    "unsupported",
    "invalid schema",
    "strict",
)


@dataclass(frozen=True)
class _ApiStructuredStrategy:
    name: str
    constraint_mode: SchemaConstraintMode


class _ApiStructuredUnavailableError(Exception):
    """Raised when a server structured-output strategy cannot be used."""


def _disable_anthropic_thinking(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep Anthropic Messages calls lean and avoid hidden extended-thinking tails."""
    payload["thinking"] = {"type": "disabled"}
    return payload


class LlmClient:
    """Text and vision client for a configured OpenAI-compatible server."""

    def __init__(self, config: LLMConfig) -> None:
        import openai

        if not config.model.strip():
            raise ValueError("Set llm.model in the configuration or VIGIL_LLM_MODEL")
        self._config = config
        self._provider = config.provider
        self._client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "not-required",
            max_retries=0,
        )
        self._model = config.model
        self._api_model_metadata_cache: dict[str, dict[str, Any]] = {}

    def _add_sampling_params(
        self, payload: dict[str, Any], endpoint: str = "/chat/completions"
    ) -> dict[str, Any]:
        """Attach optional sampling parameters only when explicitly configured."""
        if self._config.temperature is not None:
            # The /responses API rejects the temperature parameter for
            # gpt-5-family models ("Unsupported parameter").
            if endpoint == "/responses" and self._model.startswith("gpt-5"):
                return payload
            payload["temperature"] = self._config.temperature
        return payload

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Text-only generation."""
        return self._generate_api(system_prompt, user_prompt, images=None, image_labels=None)

    def generate_with_images(
        self,
        system_prompt: str,
        text_prompt: str,
        images: list[Path],
        image_labels: list[str] | None = None,
    ) -> str:
        """Multimodal generation with one or more images.

        Each image is loaded, resized to max 1280px longest edge,
        and sent as image content blocks.
        """
        return self._generate_api(
            system_prompt,
            text_prompt,
            images=images,
            image_labels=image_labels,
        )

    # ------------------------------------------------------------------
    # Structured (schema-constrained) generation
    # ------------------------------------------------------------------

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        schema_name: str,
    ) -> StructuredResult:
        """Generate a single object validated against ``response_model``.

        Uses a supported structured-output endpoint. If none is available, returns
        ``parsed=None`` with ``schema_constraint_mode="prompt_only_unavailable"``.
        """
        return self._structured_api(
            system_prompt,
            user_prompt,
            response_model,
            schema_name,
            images=None,
            image_labels=None,
        )

    def generate_structured_with_images(
        self,
        system_prompt: str,
        text_prompt: str,
        images: list[Path],
        response_model: type[BaseModel],
        schema_name: str,
        image_labels: list[str] | None = None,
    ) -> StructuredResult:
        """Multimodal :meth:`generate_structured`. Same constraint-mode semantics."""
        return self._structured_api(
            system_prompt,
            text_prompt,
            response_model,
            schema_name,
            images=images,
            image_labels=image_labels,
        )

    def _generate_api(
        self,
        system_prompt: str,
        text_prompt: str,
        *,
        images: list[Path] | None,
        image_labels: list[str] | None,
    ) -> str:
        metadata = self._api_model_metadata(self._model)
        vendor = self._api_vendor(metadata).lower()
        endpoints = set(metadata.get("supported_endpoints") or [])
        no_endpoint_metadata = not endpoints

        def has(endpoint: str) -> bool:
            return no_endpoint_metadata or endpoint in endpoints

        is_anthropic = "anthropic" in vendor or self._model.startswith("claude")
        is_openai = "openai" in vendor or "azure" in vendor or self._model.startswith("gpt")
        if is_anthropic and has("/v1/messages"):
            return self._generate_api_anthropic_messages(
                system_prompt,
                text_prompt,
                images,
                image_labels,
                metadata,
            )
        # GPT-family models exposed through the configured server use Responses as their
        # canonical plain/multimodal transport. Some metadata also advertises the chat
        # endpoint for compatibility, but selecting it first can produce a hard 400.
        if is_openai and has("/responses"):
            return self._generate_api_responses(system_prompt, text_prompt, images, image_labels)
        if has("/chat/completions"):
            if images:
                return self._generate_openai_compatible_with_images(
                    system_prompt,
                    text_prompt,
                    images,
                    image_labels,
                )
            return self._generate_openai_compatible(system_prompt, text_prompt)
        if has("/responses"):
            return self._generate_api_responses(system_prompt, text_prompt, images, image_labels)
        raise ValueError(f"Server model {self._model!r} has no supported plain generation endpoint")

    def _generate_api_anthropic_messages(
        self,
        system_prompt: str,
        text_prompt: str,
        images: list[Path] | None,
        image_labels: list[str] | None,
        metadata: dict[str, Any],
    ) -> str:
        if images:
            labels = image_labels or [None] * len(images)
            pil_images = [self._preprocess_image(p) for p in images]
            content: Any = self._anthropic_image_content(text_prompt, pil_images, labels)
        else:
            content = text_prompt
        payload = {
            "model": self._model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
            # Anthropic Messages requires this field; use the model/server output ceiling,
            # not a project budget or artificial truncation limit.
            "max_tokens": self._api_anthropic_max_tokens(metadata),
        }
        _disable_anthropic_thinking(payload)
        self._add_sampling_params(payload, endpoint="/messages")
        response = self._api_post_json("/messages", payload)
        return self._api_messages_text(response)

    def _generate_api_responses(
        self,
        system_prompt: str,
        text_prompt: str,
        images: list[Path] | None,
        image_labels: list[str] | None,
    ) -> str:
        user_content: Any = (
            self._openai_response_input_content(text_prompt, images, image_labels)
            if images
            else text_prompt
        )
        payload = {
            "model": self._model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        self._add_sampling_params(payload, endpoint="/responses")
        response = self._api_post_json("/responses", payload)
        return self._api_response_output_text(response)

    def _structured_api(
        self,
        system_prompt: str,
        text_prompt: str,
        response_model: type[BaseModel],
        schema_name: str,
        *,
        images: list[Path] | None,
        image_labels: list[str] | None,
    ) -> StructuredResult:
        metadata = self._api_model_metadata(self._model)
        failures: list[str] = []
        for strategy in self._api_strategy_order(metadata):
            try:
                result = self._run_api_strategy(
                    strategy,
                    system_prompt,
                    text_prompt,
                    response_model,
                    schema_name,
                    images,
                    image_labels,
                    metadata,
                )
            except _ApiStructuredUnavailableError as exc:
                failures.append(f"{strategy.name}: {exc}")
                continue
            return result

        reason = "; ".join(failures) or f"no structured strategy available for {self._model}"
        logger.warning(f"Structured output unavailable ({reason}); failing clearly.")
        return StructuredResult(
            parsed=None,
            raw_text="",
            provider=self._provider,
            model=self._model,
            schema_name=schema_name,
            schema_constraint_mode="prompt_only_unavailable",
            validation_errors=[reason],
        )

    def _api_model_metadata(self, model: str) -> dict[str, Any]:
        cache = self._api_model_metadata_cache
        if model in cache:
            return cache[model]

        try:
            models_url = self._config.models_url or self._api_url("/models")
            payload = self._api_get_json(models_url)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(f"Could not query server model metadata: {exc}")
            cache[model] = {}
            return {}

        data = payload.get("data", [])
        for item in data:
            if not isinstance(item, dict):
                continue
            ids = {str(item.get("id") or ""), str(item.get("claude_model_id") or "")}
            if model in ids:
                cache[model] = item
                return item
        cache[model] = {}
        return {}

    def _api_strategy_order(self, metadata: dict[str, Any]) -> list[_ApiStructuredStrategy]:
        vendor = self._api_vendor(metadata).lower()
        endpoints = set(metadata.get("supported_endpoints") or [])
        supports = (metadata.get("capabilities") or {}).get("supports") or {}
        no_endpoint_metadata = not endpoints

        def has_endpoint(endpoint: str) -> bool:
            return no_endpoint_metadata or endpoint in endpoints

        def supports_flag(flag: str) -> bool:
            return no_endpoint_metadata or bool(supports.get(flag))

        is_anthropic = "anthropic" in vendor or self._model.startswith("claude")
        is_google = "google" in vendor or self._model.startswith("gemini")
        is_openai = "openai" in vendor or "azure" in vendor or self._model.startswith("gpt")

        strategies: list[_ApiStructuredStrategy] = []

        def add(name: str, mode: SchemaConstraintMode) -> None:
            strategies.append(_ApiStructuredStrategy(name, mode))

        if is_anthropic:
            if has_endpoint("/v1/messages") and supports_flag("tool_calls"):
                add("anthropic_messages_tool", "tool_schema")
            if has_endpoint("/chat/completions") and supports_flag("tool_calls"):
                add("chat_function_tool", "tool_schema")
        elif is_google:
            if has_endpoint("/chat/completions") and supports_flag("tool_calls"):
                add("chat_function_tool", "tool_schema")
            if has_endpoint("/chat/completions"):
                add("chat_json_schema", "native_schema")
        elif is_openai:
            if has_endpoint("/responses") and supports_flag("structured_outputs"):
                add("responses_json_schema", "native_schema")
            if has_endpoint("/chat/completions") and supports_flag("structured_outputs"):
                add("chat_json_schema", "native_schema")
            if has_endpoint("/chat/completions") and supports_flag("tool_calls"):
                add("chat_function_tool", "tool_schema")
        else:
            if has_endpoint("/v1/messages") and supports_flag("tool_calls"):
                add("anthropic_messages_tool", "tool_schema")
            if has_endpoint("/responses") and supports_flag("structured_outputs"):
                add("responses_json_schema", "native_schema")
            if has_endpoint("/chat/completions") and supports_flag("tool_calls"):
                add("chat_function_tool", "tool_schema")
            if has_endpoint("/chat/completions") and supports_flag("structured_outputs"):
                add("chat_json_schema", "native_schema")

        return strategies

    def _run_api_strategy(
        self,
        strategy: _ApiStructuredStrategy,
        system_prompt: str,
        text_prompt: str,
        response_model: type[BaseModel],
        schema_name: str,
        images: list[Path] | None,
        image_labels: list[str] | None,
        metadata: dict[str, Any],
    ) -> StructuredResult:
        if strategy.name == "anthropic_messages_tool":
            return self._api_anthropic_messages_tool(
                system_prompt,
                text_prompt,
                response_model,
                schema_name,
                images,
                image_labels,
                metadata,
                strategy,
            )
        if strategy.name == "chat_function_tool":
            return self._api_chat_function_tool(
                system_prompt,
                text_prompt,
                response_model,
                schema_name,
                images,
                image_labels,
                strategy,
            )
        if strategy.name == "responses_json_schema":
            return self._api_responses_json_schema(
                system_prompt,
                text_prompt,
                response_model,
                schema_name,
                images,
                image_labels,
                strategy,
            )
        if strategy.name == "chat_json_schema":
            return self._api_chat_json_schema(
                system_prompt,
                text_prompt,
                response_model,
                schema_name,
                images,
                image_labels,
                strategy,
            )
        raise _ApiStructuredUnavailableError(f"unknown strategy {strategy.name}")

    def _api_anthropic_messages_tool(
        self,
        system_prompt: str,
        text_prompt: str,
        response_model: type[BaseModel],
        schema_name: str,
        images: list[Path] | None,
        image_labels: list[str] | None,
        metadata: dict[str, Any],
        strategy: _ApiStructuredStrategy,
    ) -> StructuredResult:
        content: Any
        if images:
            labels = image_labels or [None] * len(images)
            pil_images = [self._preprocess_image(p) for p in images]
            content = self._anthropic_image_content(text_prompt, pil_images, labels)
        else:
            content = text_prompt

        tool = {
            "name": schema_name,
            "description": f"Return exactly one well-formed {schema_name} object.",
            "input_schema": to_strict_schema(response_model, inline_refs=True),
            "strict": True,
        }
        payload = {
            "model": self._model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": schema_name},
            "max_tokens": self._api_anthropic_max_tokens(metadata),
        }
        _disable_anthropic_thinking(payload)
        self._add_sampling_params(payload, endpoint="/messages")
        response = self._api_post_json("/messages", payload)
        block = next(
            (
                b
                for b in response.get("content", [])
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ),
            None,
        )
        raw = json.dumps(block.get("input")) if isinstance(block, dict) else ""
        parsed, errors = self._validate_api_obj(
            response_model,
            block.get("input") if block else None,
        )
        stop = str(response.get("stop_reason") or "")
        return self._api_structured_result(
            parsed,
            raw,
            schema_name,
            strategy,
            refusal="refused" if stop == "refusal" else None,
            incomplete=(stop == "max_tokens"),
            incomplete_detail="max_tokens" if stop == "max_tokens" else None,
            validation_errors=errors or ([] if block else ["forced tool call was not returned"]),
        )

    def _api_chat_function_tool(
        self,
        system_prompt: str,
        text_prompt: str,
        response_model: type[BaseModel],
        schema_name: str,
        images: list[Path] | None,
        image_labels: list[str] | None,
        strategy: _ApiStructuredStrategy,
    ) -> StructuredResult:
        user_content: Any = (
            self._openai_image_content(text_prompt, images, image_labels) if images else text_prompt
        )
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": schema_name,
                        "description": f"Return exactly one well-formed {schema_name} object.",
                        "parameters": to_strict_schema(response_model, inline_refs=True),
                        "strict": True,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": schema_name}},
        }
        self._add_sampling_params(payload)
        response = self._api_post_json("/chat/completions", payload)
        choice = self._api_first_choice(response)
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        raw = self._api_tool_arguments(message)
        parsed, errors = self._validate_api_json(response_model, raw)
        finish = str(choice.get("finish_reason") or "") if isinstance(choice, dict) else ""
        return self._api_structured_result(
            parsed,
            raw,
            schema_name,
            strategy,
            refusal=message.get("refusal") if isinstance(message, dict) else None,
            incomplete=(finish == "length"),
            incomplete_detail="length" if finish == "length" else None,
            validation_errors=errors or ([] if raw else ["forced tool call was not returned"]),
        )

    def _api_responses_json_schema(
        self,
        system_prompt: str,
        text_prompt: str,
        response_model: type[BaseModel],
        schema_name: str,
        images: list[Path] | None,
        image_labels: list[str] | None,
        strategy: _ApiStructuredStrategy,
    ) -> StructuredResult:
        user_content: Any = (
            self._openai_response_input_content(text_prompt, images, image_labels)
            if images
            else text_prompt
        )
        payload = {
            "model": self._model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": to_strict_schema(response_model, inline_refs=True),
                }
            },
        }
        self._add_sampling_params(payload, endpoint="/responses")
        response = self._api_post_json("/responses", payload)
        raw = self._api_response_output_text(response)
        parsed, errors = self._validate_api_json(response_model, raw)
        mode = "native_schema" if parsed is not None else "native_schema_unenforced"
        status = str(response.get("status") or "")
        return self._api_structured_result(
            parsed,
            raw,
            schema_name,
            strategy,
            mode=mode,
            refusal=self._api_response_refusal(response),
            incomplete=(status == "incomplete"),
            incomplete_detail=str(response.get("incomplete_details") or "") or None,
            validation_errors=errors,
        )

    def _api_chat_json_schema(
        self,
        system_prompt: str,
        text_prompt: str,
        response_model: type[BaseModel],
        schema_name: str,
        images: list[Path] | None,
        image_labels: list[str] | None,
        strategy: _ApiStructuredStrategy,
    ) -> StructuredResult:
        user_content: Any = (
            self._openai_image_content(text_prompt, images, image_labels) if images else text_prompt
        )
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": to_strict_schema(response_model, inline_refs=True),
                },
            },
        }
        self._add_sampling_params(payload)
        response = self._api_post_json("/chat/completions", payload)
        choice = self._api_first_choice(response)
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        raw = str(message.get("content") or "") if isinstance(message, dict) else ""
        parsed, errors = self._validate_api_json(response_model, raw)
        mode = "native_schema" if parsed is not None else "native_schema_unenforced"
        finish = str(choice.get("finish_reason") or "") if isinstance(choice, dict) else ""
        return self._api_structured_result(
            parsed,
            raw,
            schema_name,
            strategy,
            mode=mode,
            refusal=message.get("refusal") if isinstance(message, dict) else None,
            incomplete=(finish == "length"),
            incomplete_detail="length" if finish == "length" else None,
            validation_errors=errors,
        )

    def _api_structured_result(
        self,
        parsed: BaseModel | None,
        raw: str,
        schema_name: str,
        strategy: _ApiStructuredStrategy,
        *,
        mode: SchemaConstraintMode | None = None,
        refusal: str | None = None,
        incomplete: bool = False,
        incomplete_detail: str | None = None,
        validation_errors: list[str] | None = None,
    ) -> StructuredResult:
        return StructuredResult(
            parsed=parsed,
            raw_text=raw,
            provider=self._provider,
            model=self._model,
            schema_name=schema_name,
            schema_constraint_mode=mode or strategy.constraint_mode,
            refusal=refusal,
            incomplete=incomplete,
            incomplete_detail=incomplete_detail,
            validation_errors=validation_errors or [],
        )

    def _api_get_json(self, url: str) -> dict[str, Any]:
        response = httpx.get(url, headers=self._api_headers(), timeout=10)
        response.raise_for_status()
        return response.json()

    def _api_post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._api_url(path)

        def _call() -> dict[str, Any]:
            timeout = httpx.Timeout(
                connect=10,
                read=_API_POST_READ_TIMEOUT_SECONDS,
                write=30,
                pool=10,
            )
            response = httpx.post(url, headers=self._api_headers(), json=payload, timeout=timeout)
            if response.status_code in {400, 404, 405, 422} and (
                response.status_code in {404, 405}
                or any(marker in response.text.lower() for marker in _SCHEMA_UNSUPPORTED_MARKERS)
            ):
                raise _ApiStructuredUnavailableError(
                    f"{path} structured endpoint unsupported: {response.text}"
                )
            response.raise_for_status()
            return response.json()

        return self._call_with_retry(_call)

    def _api_url(self, path: str) -> str:
        return f"{self._config.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _api_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _api_vendor(self, metadata: dict[str, Any]) -> str:
        vendor = str(metadata.get("vendor") or metadata.get("owned_by") or "")
        if vendor:
            return vendor
        if self._model.startswith("claude"):
            return "Anthropic"
        if self._model.startswith("gemini"):
            return "Google"
        if self._model.startswith("gpt"):
            return "OpenAI"
        return "unknown"

    @staticmethod
    def _api_first_choice(response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            return choices[0]
        return {}

    @staticmethod
    def _api_tool_arguments(message: dict[str, Any]) -> str:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(tool_calls, list) or not tool_calls:
            return ""
        call = tool_calls[0]
        if not isinstance(call, dict):
            return ""
        function = call.get("function")
        if not isinstance(function, dict):
            return ""
        args = function.get("arguments")
        if isinstance(args, str):
            return args
        if isinstance(args, dict):
            return json.dumps(args)
        return ""

    @staticmethod
    def _api_messages_text(response: dict[str, Any]) -> str:
        chunks: list[str] = []
        content = response.get("content")
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks)

    @staticmethod
    def _api_response_output_text(response: dict[str, Any]) -> str:
        output_text = response.get("output_text")
        if isinstance(output_text, str):
            return output_text
        chunks: list[str] = []
        for item in response.get("output", []) if isinstance(response.get("output"), list) else []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []) if isinstance(item.get("content"), list) else []:
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)

    @staticmethod
    def _api_response_refusal(response: dict[str, Any]) -> str | None:
        for item in response.get("output", []) if isinstance(response.get("output"), list) else []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []) if isinstance(item.get("content"), list) else []:
                if isinstance(content, dict) and content.get("type") == "refusal":
                    refusal = content.get("refusal") or content.get("text")
                    return str(refusal) if refusal else "refusal"
        return None

    @staticmethod
    def _validate_api_json(
        response_model: type[BaseModel],
        raw: str,
    ) -> tuple[BaseModel | None, list[str]]:
        if not raw:
            return None, ["structured response was empty"]
        try:
            return response_model.model_validate_json(raw), []
        except ValidationError as exc:
            return None, [str(exc)]

    @staticmethod
    def _validate_api_obj(
        response_model: type[BaseModel],
        obj: Any,
    ) -> tuple[BaseModel | None, list[str]]:
        if obj is None:
            return None, ["structured response was empty"]
        try:
            return response_model.model_validate(obj), []
        except ValidationError as exc:
            return None, [str(exc)]

    @staticmethod
    def _api_anthropic_max_tokens(metadata: dict[str, Any]) -> int:
        limits = (metadata.get("capabilities") or {}).get("limits") or {}
        value = limits.get("max_non_streaming_output_tokens") or limits.get("max_output_tokens")
        if isinstance(value, int) and value > 0:
            return min(value, _ANTHROPIC_NON_STREAMING_MAX_OUTPUT_TOKENS)
        return _ANTHROPIC_NON_STREAMING_MAX_OUTPUT_TOKENS

    def _openai_response_input_content(
        self,
        text_prompt: str,
        images: list[Path],
        image_labels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        labels = image_labels or [None] * len(images)
        pil_images = [self._preprocess_image(p) for p in images]
        content: list[dict[str, Any]] = []
        for label, img in zip(labels, pil_images, strict=True):
            if label:
                content.append({"type": "input_text", "text": label})
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                }
            )
        content.append({"type": "input_text", "text": text_prompt})
        return content

    def _anthropic_image_content(
        self,
        text_prompt: str,
        images: list[Image.Image],
        labels: list[str | None],
    ) -> list[dict[str, Any]]:
        """Build the Anthropic user content list (labels + base64 image blocks + text)."""
        content: list[dict[str, Any]] = []
        for label, img in zip(labels, images, strict=True):
            if label:
                content.append({"type": "text", "text": label})
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                }
            )
        content.append({"type": "text", "text": text_prompt})
        return content

    def _generate_openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            self._add_sampling_params(kwargs)
            return self._client.chat.completions.create(**kwargs)

        response = self._call_with_retry(_call)
        if response.usage:
            logger.debug(
                f"OpenAI-compatible tokens: input={response.usage.prompt_tokens}, "
                f"output={response.usage.completion_tokens}"
            )
        return self._openai_message_content(response)

    def _generate_openai_compatible_with_images(
        self,
        system_prompt: str,
        text_prompt: str,
        images: list[Path],
        image_labels: list[str] | None = None,
    ) -> str:
        """Multimodal generation via OpenAI-compatible server with base64 images."""
        content = self._openai_image_content(text_prompt, images, image_labels)

        def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
            }
            self._add_sampling_params(kwargs)
            return self._client.chat.completions.create(**kwargs)

        response = self._call_with_retry(_call)
        if response.usage:
            logger.debug(
                f"OpenAI-compatible tokens: input={response.usage.prompt_tokens}, "
                f"output={response.usage.completion_tokens}"
            )
        return self._openai_message_content(response)

    def _openai_image_content(
        self,
        text_prompt: str,
        images: list[Path],
        image_labels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the OpenAI-compatible user content list (labels + base64 image_url blocks)."""
        labels = image_labels or [None] * len(images)
        pil_images = [self._preprocess_image(p) for p in images]
        content: list[dict[str, Any]] = []
        for label, img in zip(labels, pil_images, strict=True):
            if label:
                content.append({"type": "text", "text": label})
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        content.append({"type": "text", "text": text_prompt})
        return content

    @staticmethod
    def _openai_message_content(response: Any) -> str:
        if not response.choices:
            raise ValueError("LLM response contained no choices")
        return response.choices[0].message.content or ""

    @staticmethod
    def _preprocess_image(path: Path) -> Image.Image:
        """Load and resize image so longest edge ≤ 1280px."""
        img = Image.open(path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > _MAX_IMAGE_EDGE:
            scale = _MAX_IMAGE_EDGE / longest
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    def _call_with_retry(self, fn: Callable[[], Any]) -> Any:
        """Call fn with exponential backoff retry on transient errors."""
        for attempt in range(_MAX_RETRIES):
            try:
                return fn()
            except Exception as exc:
                if attempt == _MAX_RETRIES - 1 or not self._is_retryable(exc):
                    raise
                backoff = _BACKOFF_BASE * (2**attempt)
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{_MAX_RETRIES}), "
                    f"retrying in {backoff}s: {exc}"
                )
                time.sleep(backoff)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Check if an exception is transient and should be retried."""
        from openai import APIConnectionError, APIStatusError

        if isinstance(exc, httpx.TimeoutException | httpx.NetworkError):
            return True
        if isinstance(exc, APIConnectionError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {429, 500, 502, 503}
        if isinstance(exc, APIStatusError):
            return exc.status_code in {429, 500, 502, 503}
        return False

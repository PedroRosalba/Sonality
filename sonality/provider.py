from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import config

log = logging.getLogger(__name__)

_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_EMPTY_MAPPING: dict[str, object] = {}


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Normalized chat completion payload used by all model call sites."""

    text: str
    input_tokens: int
    output_tokens: int
    raw: dict[str, object]


def _to_nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    return 0


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.API_KEY:
        headers["Authorization"] = f"Bearer {config.API_KEY}"
    return headers


def _endpoint(path: str) -> str:
    base = config.BASE_URL.rstrip("/")
    normalized = path if path.startswith("/") else f"/{path}"
    return f"{base}{normalized}"


def _post_json(path: str, payload: Mapping[str, object]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        request = Request(
            _endpoint(path),
            data=body,
            headers=_headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                decoded = response.read().decode("utf-8")
            parsed = json.loads(decoded)
            if isinstance(parsed, dict):
                return parsed
            raise RuntimeError("Provider returned a non-object JSON payload")
        except HTTPError as exc:
            status = int(exc.code)
            if attempt < max_attempts and status in _RETRYABLE_HTTP_STATUSES:
                wait = 1.5**attempt
                log.warning(
                    "Provider HTTP %d on attempt %d/%d; retrying in %.1fs",
                    status,
                    attempt,
                    max_attempts,
                    wait,
                )
                time.sleep(wait)
                continue
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = "<failed to read error payload>"
            raise RuntimeError(f"Provider HTTP {status}: {detail}") from exc
        except URLError as exc:
            if attempt < max_attempts:
                wait = 1.5**attempt
                log.warning(
                    "Provider network error on attempt %d/%d; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"Provider network error: {exc}") from exc
        except (TimeoutError, ConnectionError, OSError) as exc:
            if attempt < max_attempts:
                wait = 1.5**attempt
                log.warning(
                    "Provider transport error on attempt %d/%d; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"Provider transport error: {exc}") from exc
    raise RuntimeError("Provider request failed after retries")


def _message_content_text(message: object) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, list):
        return ""
    parts: list[str] = []
    for item in message:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def chat_completion(
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    max_tokens: int,
    temperature: float = -1.0,
    tools: Sequence[Mapping[str, object]] = (),
    tool_choice: Mapping[str, object] = _EMPTY_MAPPING,
) -> ChatResult:
    payload: dict[str, object] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "max_tokens": max_tokens,
    }
    if temperature >= 0.0:
        payload["temperature"] = temperature
    if tools:
        payload["tools"] = [dict(tool) for tool in tools]
    if tool_choice:
        payload["tool_choice"] = dict(tool_choice)

    raw = _post_json("/chat/completions", payload)
    text = ""
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                text = _message_content_text(message.get("content", ""))
    usage = raw.get("usage")
    input_tokens = 0
    output_tokens = 0
    if isinstance(usage, dict):
        input_tokens = _to_nonnegative_int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
        output_tokens = _to_nonnegative_int(
            usage.get("completion_tokens", usage.get("output_tokens", 0))
        )
    return ChatResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw=raw,
    )


def embed(
    *,
    model: str,
    texts: list[str],
    dimensions: int = 0,
) -> list[list[float]]:
    payload: dict[str, object] = {"model": model, "input": texts}
    if dimensions > 0:
        payload["dimensions"] = dimensions
    raw = _post_json("/embeddings", payload)
    data = raw.get("data")
    if not isinstance(data, list):
        raise RuntimeError("Embedding response is missing `data` array")
    sorted_rows = sorted(
        (row for row in data if isinstance(row, dict)),
        key=lambda row: int(row.get("index", 0)),
    )
    vectors: list[list[float]] = []
    for row in sorted_rows:
        embedding = row.get("embedding")
        if not isinstance(embedding, list):
            continue
        vectors.append([float(v) for v in embedding])
    return vectors


def parse_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if not stripped:
        return {}
    for candidate in (
        stripped,
        stripped.replace("```json", "").replace("```", "").strip(),
    ):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    block_match = _JSON_BLOCK_RE.search(stripped)
    if block_match is None:
        return {}
    try:
        parsed = json.loads(block_match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_tool_call_arguments(
    raw_payload: Mapping[str, object], function_name: str
) -> dict[str, object]:
    choices = raw_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return {}
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return {}
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return {}
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name") != function_name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            return dict(arguments)
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return dict(parsed)
    return {}

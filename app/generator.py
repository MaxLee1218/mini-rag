from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 30.0
DEFAULT_THINKING_TYPE = "disabled"
MAX_ERROR_BODY_CHARS = 500


class GeneratorError(Exception):
    """Base exception for generator errors."""


class MissingAPIKeyError(GeneratorError):
    """Raised when DEEPSEEK_API_KEY is missing."""


class DeepSeekAPIError(GeneratorError):
    """Raised when DeepSeek API returns an error or invalid response."""


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    model: str = DEFAULT_DEEPSEEK_MODEL
    timeout: float = DEFAULT_TIMEOUT


def load_deepseek_config_from_env() -> DeepSeekConfig:
    """Load DeepSeek configuration from system environment variables."""
    import app.config as app_config

    try:
        api_key = app_config.require_deepseek_api_key()
    except RuntimeError as error:
        raise MissingAPIKeyError(str(error)) from error

    return _validate_config(
        DeepSeekConfig(
            api_key=api_key,
            base_url=app_config.DEEPSEEK_BASE_URL,
            model=app_config.DEEPSEEK_MODEL,
            timeout=app_config.DEEPSEEK_TIMEOUT,
        )
    )


def generate_answer(
    prompt: str,
    *,
    config: DeepSeekConfig | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Send a prompt to DeepSeek and return the final answer text."""
    clean_prompt = _validate_prompt(prompt)
    resolved_config = _validate_config(
        load_deepseek_config_from_env() if config is None else config
    )
    clean_temperature = _validate_temperature(temperature)
    clean_max_tokens = _validate_max_tokens(max_tokens)

    response = _post_chat_completion(
        prompt=clean_prompt,
        config=resolved_config,
        temperature=clean_temperature,
        max_tokens=clean_max_tokens,
    )
    return _extract_answer(response)


class DeepSeekGenerator:
    def __init__(
        self,
        config: DeepSeekConfig | None = None,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.config = config
        self.temperature = _validate_temperature(temperature)
        self.max_tokens = _validate_max_tokens(max_tokens)

    def generate(self, prompt: str) -> str:
        """Generate an answer for a complete RAG prompt."""
        return generate_answer(
            prompt,
            config=self.config,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


def _post_chat_completion(
    *,
    prompt: str,
    config: DeepSeekConfig,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    url = f"{config.base_url}/chat/completions"
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "thinking": {"type": DEFAULT_THINKING_TYPE},
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            response_body = response.read()
    except urllib.error.HTTPError as error:
        raise _http_error(config, error) from error
    except urllib.error.URLError as error:
        message = _redact_api_key(f"DeepSeek API request failed: {error}", config)
        raise DeepSeekAPIError(message) from error
    except (TimeoutError, socket.timeout) as error:
        message = _redact_api_key(f"DeepSeek API request timed out: {error}", config)
        raise DeepSeekAPIError(message) from error

    try:
        decoded = response_body.decode("utf-8")
        data = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        message = _redact_api_key("DeepSeek API returned invalid JSON", config)
        raise DeepSeekAPIError(message) from error

    if not isinstance(data, dict):
        raise DeepSeekAPIError("DeepSeek API returned a non-object response")
    return data


def _extract_answer(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise DeepSeekAPIError("DeepSeek API response is missing answer content") from error

    if not isinstance(content, str):
        raise DeepSeekAPIError("DeepSeek API answer content must be a string")

    answer = content.strip()
    if not answer:
        raise DeepSeekAPIError("DeepSeek API answer content is empty")
    return answer


def _validate_config(config: DeepSeekConfig) -> DeepSeekConfig:
    if not isinstance(config, DeepSeekConfig):
        raise ValueError("config must be a DeepSeekConfig")

    api_key = _validate_nonblank_string(config.api_key, "api_key")
    base_url = _strip_base_url(_validate_nonblank_string(config.base_url, "base_url"))
    model = _validate_nonblank_string(config.model, "model")
    timeout = _validate_timeout(config.timeout)

    return DeepSeekConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
    )


def _validate_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if not prompt.strip():
        raise ValueError("prompt must not be blank")
    return prompt.strip()


def _validate_temperature(temperature: float) -> float:
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise ValueError("temperature must be a number")
    value = float(temperature)
    if value < 0 or value > 2:
        raise ValueError("temperature must be between 0 and 2")
    return value


def _validate_max_tokens(max_tokens: int) -> int:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("max_tokens must be a positive integer")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    return max_tokens


def _validate_timeout(timeout: Any) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, Real):
        raise ValueError("timeout must be a positive number")
    value = float(timeout)
    if value <= 0:
        raise ValueError("timeout must be a positive number")
    return value


def _validate_nonblank_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    clean_value = value.strip()
    if not clean_value:
        raise ValueError(f"{field_name} must not be blank")
    return clean_value


def _strip_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def _http_error(config: DeepSeekConfig, error: urllib.error.HTTPError) -> DeepSeekAPIError:
    body = ""
    try:
        body = error.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""

    body = body[:MAX_ERROR_BODY_CHARS]
    message = f"DeepSeek API HTTP {error.code}: {error.reason}"
    if body:
        message = f"{message}: {body}"
    return DeepSeekAPIError(_redact_api_key(message, config))


def _redact_api_key(message: str, config: DeepSeekConfig) -> str:
    api_key = config.api_key if isinstance(config.api_key, str) else ""
    if api_key:
        return message.replace(api_key, "[REDACTED]")
    return message

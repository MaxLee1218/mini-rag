import io
import importlib
import json
import socket
import urllib.error

import pytest

import app.generator as generator_module
from app.generator import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_TYPE,
    DEFAULT_TIMEOUT,
    DeepSeekAPIError,
    DeepSeekConfig,
    DeepSeekGenerator,
    MissingAPIKeyError,
    generate_answer,
    load_deepseek_config_from_env,
)


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        if isinstance(self.body, bytes):
            return self.body
        return json.dumps(self.body).encode("utf-8")


class UnreadableHTTPError(urllib.error.HTTPError):
    def read(self, *args, **kwargs):
        raise OSError("cannot read body")


def fake_config(**overrides):
    values = {
        "api_key": "test-api-key",
        "base_url": "https://api.test",
        "model": "test-model",
        "timeout": 3.5,
    }
    values.update(overrides)
    return DeepSeekConfig(**values)


def install_fake_urlopen(monkeypatch, body):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(body)

    monkeypatch.setattr(generator_module.urllib.request, "urlopen", fake_urlopen)
    return captured


def request_payload(request):
    return json.loads(request.data.decode("utf-8"))


def request_headers(request):
    return {key.lower(): value for key, value in request.header_items()}


def reload_app_config(monkeypatch):
    import app.config as config

    return importlib.reload(config)


def test_load_deepseek_config_from_env_reads_values_and_trims_base_url(monkeypatch):
    monkeypatch.setenv("MINI_RAG_SKIP_DOTENV", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.delenv("DEEPSEEK_TIMEOUT", raising=False)
    reload_app_config(monkeypatch)

    config = load_deepseek_config_from_env()

    assert config.api_key == "test-key"
    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-v4-flash"
    assert config.timeout == DEFAULT_TIMEOUT


def test_load_deepseek_config_from_env_uses_default_model(monkeypatch):
    monkeypatch.setenv("MINI_RAG_SKIP_DOTENV", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_TIMEOUT", raising=False)
    reload_app_config(monkeypatch)

    config = load_deepseek_config_from_env()

    assert config.base_url == DEFAULT_DEEPSEEK_BASE_URL
    assert config.model == DEFAULT_DEEPSEEK_MODEL


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_load_deepseek_config_from_env_rejects_missing_api_key(monkeypatch, api_key):
    monkeypatch.setenv("MINI_RAG_SKIP_DOTENV", "1")
    if api_key is None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    else:
        monkeypatch.setenv("DEEPSEEK_API_KEY", api_key)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_TIMEOUT", raising=False)
    reload_app_config(monkeypatch)

    with pytest.raises(MissingAPIKeyError):
        load_deepseek_config_from_env()


def test_generate_answer_returns_content_and_sends_expected_payload(monkeypatch):
    captured = install_fake_urlopen(
        monkeypatch,
        {"choices": [{"message": {"content": "这是最终回答。"}}]},
    )
    config = fake_config()

    answer = generate_answer("prompt", config=config)

    assert answer == "这是最终回答。"
    request = captured["request"]
    assert request.full_url == "https://api.test/chat/completions"
    assert captured["timeout"] == 3.5
    headers = request_headers(request)
    assert headers["authorization"] == "Bearer test-api-key"
    assert headers["content-type"] == "application/json"
    assert request_payload(request) == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "prompt"}],
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "stream": False,
        "thinking": {"type": DEFAULT_THINKING_TYPE},
    }


def test_generate_answer_uses_custom_model_from_config(monkeypatch):
    captured = install_fake_urlopen(
        monkeypatch,
        {"choices": [{"message": {"content": "answer"}}]},
    )

    generate_answer("prompt", config=fake_config(model="custom-test-model"))

    assert request_payload(captured["request"])["model"] == "custom-test-model"


def test_generate_answer_sends_custom_temperature_and_max_tokens(monkeypatch):
    captured = install_fake_urlopen(
        monkeypatch,
        {"choices": [{"message": {"content": "answer"}}]},
    )

    generate_answer(
        "prompt",
        config=fake_config(),
        temperature=0.7,
        max_tokens=256,
    )

    payload = request_payload(captured["request"])
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 256


def test_generate_answer_rejects_empty_content(monkeypatch):
    install_fake_urlopen(
        monkeypatch,
        {"choices": [{"message": {"content": "   "}}]},
    )

    with pytest.raises(DeepSeekAPIError):
        generate_answer("prompt", config=fake_config())


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
    ],
)
def test_generate_answer_rejects_malformed_response(monkeypatch, response):
    install_fake_urlopen(monkeypatch, response)

    with pytest.raises(DeepSeekAPIError):
        generate_answer("prompt", config=fake_config())


def test_generate_answer_wraps_http_error_and_redacts_api_key(monkeypatch):
    body = ("secret-key " + "x" * 700).encode("utf-8")
    error = urllib.error.HTTPError(
        url="https://api.test/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=io.BytesIO(body),
    )

    def fake_urlopen(request, timeout):
        raise error

    monkeypatch.setattr(generator_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DeepSeekAPIError) as error_info:
        generate_answer("prompt", config=fake_config(api_key="secret-key"))

    message = str(error_info.value)
    assert "HTTP 401" in message
    assert "secret-key" not in message
    assert "[REDACTED]" in message
    assert len(message) < 700


def test_generate_answer_handles_unreadable_http_error_body(monkeypatch):
    error = UnreadableHTTPError(
        url="https://api.test/chat/completions",
        code=503,
        msg="Service Unavailable",
        hdrs={},
        fp=None,
    )

    def fake_urlopen(request, timeout):
        raise error

    monkeypatch.setattr(generator_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DeepSeekAPIError, match="HTTP 503"):
        generate_answer("prompt", config=fake_config())


def test_generate_answer_decodes_non_utf8_http_error_body_safely(monkeypatch):
    error = urllib.error.HTTPError(
        url="https://api.test/chat/completions",
        code=500,
        msg="Server Error",
        hdrs={},
        fp=io.BytesIO(b"\xff\xfe"),
    )

    def fake_urlopen(request, timeout):
        raise error

    monkeypatch.setattr(generator_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DeepSeekAPIError, match="HTTP 500"):
        generate_answer("prompt", config=fake_config())


def test_generate_answer_wraps_url_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(generator_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DeepSeekAPIError):
        generate_answer("prompt", config=fake_config())


def test_generate_answer_wraps_timeout(monkeypatch):
    def fake_urlopen(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(generator_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DeepSeekAPIError):
        generate_answer("prompt", config=fake_config())


def test_generate_answer_wraps_socket_timeout(monkeypatch):
    def fake_urlopen(request, timeout):
        raise socket.timeout("timed out")

    monkeypatch.setattr(generator_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DeepSeekAPIError):
        generate_answer("prompt", config=fake_config())


def test_generate_answer_wraps_invalid_json(monkeypatch):
    install_fake_urlopen(monkeypatch, b"not-json")

    with pytest.raises(DeepSeekAPIError):
        generate_answer("prompt", config=fake_config())


@pytest.mark.parametrize("prompt", ["", "   ", None])
def test_generate_answer_rejects_invalid_prompt(prompt):
    with pytest.raises(ValueError, match="prompt must not be blank|prompt must be a string"):
        generate_answer(prompt, config=fake_config())


@pytest.mark.parametrize("temperature", [-0.1, 2.1, True, "0"])
def test_generate_answer_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        generate_answer("prompt", config=fake_config(), temperature=temperature)


@pytest.mark.parametrize("max_tokens", [0, -1, True, 1.5, "1024"])
def test_generate_answer_rejects_invalid_max_tokens(max_tokens):
    with pytest.raises(ValueError, match="max_tokens"):
        generate_answer("prompt", config=fake_config(), max_tokens=max_tokens)


@pytest.mark.parametrize(
    "config",
    [
        DeepSeekConfig(api_key="", base_url="https://api.test", model="test-model"),
        DeepSeekConfig(api_key="key", base_url="", model="test-model"),
        DeepSeekConfig(api_key="key", base_url="https://api.test", model=""),
        DeepSeekConfig(
            api_key="key",
            base_url="https://api.test",
            model="test-model",
            timeout=0,
        ),
        DeepSeekConfig(
            api_key="key",
            base_url="https://api.test",
            model="test-model",
            timeout=True,
        ),
    ],
)
def test_generate_answer_rejects_invalid_config(config):
    with pytest.raises(ValueError):
        generate_answer("prompt", config=config)


def test_generate_answer_loads_config_from_env_when_config_is_none(
    monkeypatch,
):
    monkeypatch.setenv("MINI_RAG_SKIP_DOTENV", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.test/")
    monkeypatch.setenv("DEEPSEEK_MODEL", "env-model")
    monkeypatch.setenv("DEEPSEEK_TIMEOUT", "4.5")
    reload_app_config(monkeypatch)
    captured = install_fake_urlopen(
        monkeypatch,
        {"choices": [{"message": {"content": "env answer"}}]},
    )

    answer = generate_answer("prompt")

    assert answer == "env answer"
    assert captured["request"].full_url == "https://env.test/chat/completions"
    assert captured["timeout"] == 4.5
    assert request_payload(captured["request"])["model"] == "env-model"


def test_deepseek_generator_class_generates_answer(monkeypatch):
    install_fake_urlopen(
        monkeypatch,
        {"choices": [{"message": {"content": "class answer"}}]},
    )
    generator = DeepSeekGenerator(config=fake_config())

    assert generator.generate("prompt") == "class answer"

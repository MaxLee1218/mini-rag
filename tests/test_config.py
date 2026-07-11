import importlib

import pytest


CONFIG_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_TIMEOUT",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
)


def reload_config(monkeypatch, **env_values):
    monkeypatch.setenv("MINI_RAG_SKIP_DOTENV", "1")
    for key in CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env_values.items():
        monkeypatch.setenv(key, value)

    import app.config as config

    return importlib.reload(config)


def test_config_uses_defaults_without_api_keys(monkeypatch):
    config = reload_config(monkeypatch)

    assert config.VECTOR_DB_PATH == "data/chroma"
    assert config.VECTOR_COLLECTION_NAME == "mini_rag_chunks"
    assert config.DEEPSEEK_API_KEY is None
    assert config.DEEPSEEK_BASE_URL == "https://api.deepseek.com"
    assert config.DEEPSEEK_MODEL == "deepseek-v4-flash"
    assert config.DEEPSEEK_TIMEOUT == 30.0
    assert config.OPENAI_API_KEY is None
    assert config.OPENAI_MODEL == "gpt-4o-mini"
    assert config.HYBRID_SPARSE_WEIGHT == 0.5
    assert config.HYBRID_DENSE_WEIGHT == 0.5
    assert config.HYBRID_TOP_K == 5
    assert config.HYBRID_CANDIDATE_MULTIPLIER == 2


def test_config_reads_environment_overrides(monkeypatch):
    config = reload_config(
        monkeypatch,
        DEEPSEEK_API_KEY="deepseek-test-key",
        DEEPSEEK_BASE_URL="https://deepseek.test",
        DEEPSEEK_MODEL="deepseek-test-model",
        DEEPSEEK_TIMEOUT="12.5",
        OPENAI_API_KEY="openai-test-key",
        OPENAI_MODEL="openai-test-model",
    )

    assert config.DEEPSEEK_API_KEY == "deepseek-test-key"
    assert config.DEEPSEEK_BASE_URL == "https://deepseek.test"
    assert config.DEEPSEEK_MODEL == "deepseek-test-model"
    assert config.DEEPSEEK_TIMEOUT == 12.5
    assert config.OPENAI_API_KEY == "openai-test-key"
    assert config.OPENAI_MODEL == "openai-test-model"


def test_config_rejects_invalid_deepseek_timeout(monkeypatch):
    with pytest.raises(RuntimeError, match="DEEPSEEK_TIMEOUT must be a valid number"):
        reload_config(monkeypatch, DEEPSEEK_TIMEOUT="not-a-number")


def test_require_deepseek_api_key_missing(monkeypatch):
    config = reload_config(monkeypatch)

    with pytest.raises(RuntimeError, match="Missing DEEPSEEK_API_KEY"):
        config.require_deepseek_api_key()


def test_require_deepseek_api_key_present(monkeypatch):
    config = reload_config(monkeypatch, DEEPSEEK_API_KEY="test-key")

    assert config.require_deepseek_api_key() == "test-key"


def test_require_openai_api_key_missing(monkeypatch):
    config = reload_config(monkeypatch)

    with pytest.raises(RuntimeError, match="Missing OPENAI_API_KEY"):
        config.require_openai_api_key()


def test_require_openai_api_key_present(monkeypatch):
    config = reload_config(monkeypatch, OPENAI_API_KEY="test-openai-key")

    assert config.require_openai_api_key() == "test-openai-key"

import importlib

import pytest


CONFIG_ENV_KEYS = (
    "RAG_CHUNK_MODE",
    "RAG_PARENT_CHUNK_SIZE",
    "RAG_PARENT_CHUNK_OVERLAP",
    "RAG_CHILD_CHUNK_SIZE",
    "RAG_CHILD_CHUNK_OVERLAP",
    "RAG_PARENT_STORE_PATH",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_TIMEOUT",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "RERANKER_ENABLED",
    "RERANKER_MODEL",
    "RERANKER_TOP_K",
    "RERANKER_CANDIDATE_K",
    "RERANKER_BATCH_SIZE",
    "RERANKER_MAX_LENGTH",
    "RERANKER_DEVICE",
    "RERANKER_FAILURE_MODE",
    "RERANKER_LOCAL_FILES_ONLY",
    "CONVERSATION_HISTORY_LIMIT",
    "QUERY_REWRITE_ENABLED",
    "QUERY_REWRITE_PROVIDER",
    "QUERY_REWRITE_TIMEOUT",
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
    assert config.DEFAULT_TOP_K == 5
    assert config.CHUNK_MODE == "standard"
    assert config.PARENT_CHUNK_SIZE == 1000
    assert config.PARENT_CHUNK_OVERLAP == 100
    assert config.CHILD_CHUNK_SIZE == 250
    assert config.CHILD_CHUNK_OVERLAP == 50
    assert config.PARENT_STORE_PATH == "data/parents/parents.sqlite3"
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
    assert config.CONVERSATION_HISTORY_LIMIT == 5
    assert config.QUERY_REWRITE_ENABLED is True
    assert config.QUERY_REWRITE_PROVIDER == "deepseek"
    assert config.QUERY_REWRITE_TIMEOUT == 10.0


def test_parent_child_config_reads_and_validates_environment(monkeypatch):
    config = reload_config(
        monkeypatch,
        RAG_CHUNK_MODE="parent-child",
        RAG_PARENT_CHUNK_SIZE="800",
        RAG_PARENT_CHUNK_OVERLAP="80",
        RAG_CHILD_CHUNK_SIZE="200",
        RAG_CHILD_CHUNK_OVERLAP="20",
        RAG_PARENT_STORE_PATH="tmp/parents.sqlite3",
    )

    assert config.CHUNK_MODE == "parent-child"
    assert config.PARENT_CHUNK_SIZE == 800
    assert config.CHILD_CHUNK_SIZE == 200
    assert config.PARENT_STORE_PATH == "tmp/parents.sqlite3"


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"RAG_CHUNK_MODE": "other"}, "RAG_CHUNK_MODE must be one of"),
        ({"RAG_PARENT_CHUNK_SIZE": "0"}, "RAG_PARENT_CHUNK_SIZE"),
        (
            {"RAG_PARENT_CHUNK_SIZE": "100", "RAG_CHILD_CHUNK_SIZE": "101"},
            "RAG_CHILD_CHUNK_SIZE must be smaller",
        ),
        (
            {
                "RAG_PARENT_CHUNK_SIZE": "100",
                "RAG_CHILD_CHUNK_SIZE": "50",
                "RAG_PARENT_CHUNK_OVERLAP": "100",
            },
            "RAG_PARENT_CHUNK_OVERLAP must be smaller",
        ),
        (
            {"RAG_CHILD_CHUNK_SIZE": "50", "RAG_CHILD_CHUNK_OVERLAP": "50"},
            "RAG_CHILD_CHUNK_OVERLAP must be smaller",
        ),
    ],
)
def test_parent_child_config_rejects_invalid_values(monkeypatch, env, message):
    with pytest.raises(RuntimeError, match=message):
        reload_config(monkeypatch, **env)


def test_query_rewrite_config_reads_environment_overrides(monkeypatch):
    config = reload_config(
        monkeypatch,
        CONVERSATION_HISTORY_LIMIT="3",
        QUERY_REWRITE_ENABLED="false",
        QUERY_REWRITE_PROVIDER="deepseek",
        QUERY_REWRITE_TIMEOUT="2.5",
    )

    assert config.CONVERSATION_HISTORY_LIMIT == 3
    assert config.QUERY_REWRITE_ENABLED is False
    assert config.QUERY_REWRITE_PROVIDER == "deepseek"
    assert config.QUERY_REWRITE_TIMEOUT == 2.5


@pytest.mark.parametrize("value", ["2", "6", "not-an-int"])
def test_query_rewrite_config_rejects_invalid_history_limit(monkeypatch, value):
    with pytest.raises(RuntimeError, match="CONVERSATION_HISTORY_LIMIT must be"):
        reload_config(monkeypatch, CONVERSATION_HISTORY_LIMIT=value)


def test_query_rewrite_config_rejects_unsupported_provider(monkeypatch):
    with pytest.raises(RuntimeError, match="QUERY_REWRITE_PROVIDER must be one of"):
        reload_config(monkeypatch, QUERY_REWRITE_PROVIDER="rule_based")


@pytest.mark.parametrize("value", ["0", "-1"])
def test_query_rewrite_config_rejects_non_positive_timeout(monkeypatch, value):
    with pytest.raises(RuntimeError, match="QUERY_REWRITE_TIMEOUT must be positive"):
        reload_config(monkeypatch, QUERY_REWRITE_TIMEOUT=value)


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


def test_reranker_config_defaults(monkeypatch):
    config = reload_config(monkeypatch)

    assert config.RERANKER_ENABLED is True
    assert config.RERANKER_MODEL == "cross-encoder/ms-marco-TinyBERT-L2-v2"
    assert config.RERANKER_TOP_K == 5
    assert config.RERANKER_CANDIDATE_K == 10
    assert config.RERANKER_BATCH_SIZE == 16
    assert config.RERANKER_MAX_LENGTH == 256
    assert config.RERANKER_DEVICE == "cpu"
    assert config.RERANKER_FAILURE_MODE == "fallback"
    assert config.RERANKER_LOCAL_FILES_ONLY is False


def test_reranker_config_reads_local_model_and_options(monkeypatch):
    config = reload_config(
        monkeypatch,
        RERANKER_ENABLED="false",
        RERANKER_MODEL="./models/ms-marco-TinyBERT-L2-v2",
        RERANKER_TOP_K="7",
        RERANKER_CANDIDATE_K="12",
        RERANKER_BATCH_SIZE="8",
        RERANKER_MAX_LENGTH="128",
        RERANKER_DEVICE="auto",
        RERANKER_LOCAL_FILES_ONLY="true",
    )

    assert config.RERANKER_ENABLED is False
    assert config.RERANKER_MODEL == "./models/ms-marco-TinyBERT-L2-v2"
    assert config.RERANKER_TOP_K == 7
    assert config.RERANKER_CANDIDATE_K == 12
    assert config.RERANKER_BATCH_SIZE == 8
    assert config.RERANKER_MAX_LENGTH == 128
    assert config.RERANKER_DEVICE == "auto"
    assert config.RERANKER_LOCAL_FILES_ONLY is True


@pytest.mark.parametrize(
    "name",
    [
        "RERANKER_TOP_K",
        "RERANKER_CANDIDATE_K",
        "RERANKER_BATCH_SIZE",
        "RERANKER_MAX_LENGTH",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_reranker_config_rejects_invalid_positive_integers(monkeypatch, name, value):
    with pytest.raises(RuntimeError, match=rf"{name} must be a positive integer"):
        reload_config(monkeypatch, **{name: value})


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("RERANKER_ENABLED", "perhaps", "RERANKER_ENABLED must be a boolean"),
        ("RERANKER_DEVICE", "tpu", "RERANKER_DEVICE must be one of"),
        ("RERANKER_FAILURE_MODE", "raise", "RERANKER_FAILURE_MODE must be one of"),
        ("RERANKER_MODEL", "   ", "RERANKER_MODEL must not be blank"),
    ],
)
def test_reranker_config_rejects_invalid_values(monkeypatch, name, value, message):
    with pytest.raises(RuntimeError, match=message):
        reload_config(monkeypatch, **{name: value})

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
    "FAQ_ENABLED",
    "FAQ_DB_PATH",
    "FAQ_MATCH_THRESHOLD",
    "FAQ_MATCH_MARGIN",
    "FAQ_CACHE_ENABLED",
    "FAQ_CACHE_TTL_SECONDS",
    "FAQ_CACHE_PREWARM",
    "REDIS_URL",
    "REDIS_CONNECT_TIMEOUT_SECONDS",
    "REDIS_SOCKET_TIMEOUT_SECONDS",
    "EVALUATION_DATASET_PATH",
    "EVALUATION_JSON_REPORT_PATH",
    "EVALUATION_MARKDOWN_REPORT_PATH",
    "EVALUATION_TOP_K",
    "EVALUATION_RAGAS_MODEL",
    "EVALUATION_RAGAS_EMBEDDING_MODEL",
    "EVALUATION_RAGAS_TIMEOUT",
    "EVALUATION_FAITHFULNESS_THRESHOLD",
    "EVALUATION_CONTEXT_RECALL_THRESHOLD",
    "EVALUATION_QUESTION_PREVIEW_CHARS",
    "EVALUATION_ANSWER_PREVIEW_CHARS",
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
    assert config.FAQ_ENABLED is True
    assert config.FAQ_DB_PATH == config.PROJECT_ROOT / "data/faq.db"
    assert config.FAQ_MATCH_THRESHOLD == 1.0
    assert config.FAQ_MATCH_MARGIN == 0.15
    assert config.FAQ_CACHE_ENABLED is True
    assert config.FAQ_CACHE_TTL_SECONDS == 86400
    assert config.FAQ_CACHE_PREWARM is True
    assert config.REDIS_URL == "redis://localhost:6379/0"
    assert config.REDIS_CONNECT_TIMEOUT_SECONDS == 0.2
    assert config.REDIS_SOCKET_TIMEOUT_SECONDS == 0.2
    assert config.EVALUATION_DATASET_PATH == (
        config.PROJECT_ROOT / "evaluation/dataset/eval_dataset.json"
    )
    assert config.EVALUATION_JSON_REPORT_PATH == (
        config.PROJECT_ROOT / "evaluation/reports/evaluation_report.json"
    )
    assert config.EVALUATION_MARKDOWN_REPORT_PATH == (
        config.PROJECT_ROOT / "evaluation/reports/evaluation_report.md"
    )
    assert config.EVALUATION_TOP_K == 5
    assert config.EVALUATION_RAGAS_MODEL == "gpt-4o-mini"
    assert config.EVALUATION_RAGAS_EMBEDDING_MODEL == "text-embedding-3-small"
    assert config.EVALUATION_RAGAS_TIMEOUT == 60.0
    assert config.EVALUATION_FAITHFULNESS_THRESHOLD == 0.7
    assert config.EVALUATION_CONTEXT_RECALL_THRESHOLD == 0.7
    assert config.EVALUATION_QUESTION_PREVIEW_CHARS == 300
    assert config.EVALUATION_ANSWER_PREVIEW_CHARS == 800


def test_evaluation_config_reads_environment_overrides(monkeypatch):
    config = reload_config(
        monkeypatch,
        EVALUATION_DATASET_PATH="fixtures/eval.json",
        EVALUATION_JSON_REPORT_PATH="artifacts/eval.json",
        EVALUATION_MARKDOWN_REPORT_PATH="artifacts/eval.md",
        EVALUATION_TOP_K="7",
        EVALUATION_RAGAS_MODEL="evaluation-llm",
        EVALUATION_RAGAS_EMBEDDING_MODEL="evaluation-embedding",
        EVALUATION_RAGAS_TIMEOUT="12.5",
        EVALUATION_FAITHFULNESS_THRESHOLD="0.8",
        EVALUATION_CONTEXT_RECALL_THRESHOLD="0.9",
        EVALUATION_QUESTION_PREVIEW_CHARS="120",
        EVALUATION_ANSWER_PREVIEW_CHARS="450",
    )

    assert config.EVALUATION_DATASET_PATH == config.PROJECT_ROOT / "fixtures/eval.json"
    assert config.EVALUATION_JSON_REPORT_PATH == config.PROJECT_ROOT / "artifacts/eval.json"
    assert config.EVALUATION_MARKDOWN_REPORT_PATH == config.PROJECT_ROOT / "artifacts/eval.md"
    assert config.EVALUATION_TOP_K == 7
    assert config.EVALUATION_RAGAS_MODEL == "evaluation-llm"
    assert config.EVALUATION_RAGAS_EMBEDDING_MODEL == "evaluation-embedding"
    assert config.EVALUATION_RAGAS_TIMEOUT == 12.5
    assert config.EVALUATION_FAITHFULNESS_THRESHOLD == 0.8
    assert config.EVALUATION_CONTEXT_RECALL_THRESHOLD == 0.9
    assert config.EVALUATION_QUESTION_PREVIEW_CHARS == 120
    assert config.EVALUATION_ANSWER_PREVIEW_CHARS == 450


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("EVALUATION_DATASET_PATH", " ", "EVALUATION_DATASET_PATH must not be blank"),
        ("EVALUATION_TOP_K", "0", "EVALUATION_TOP_K must be a positive integer"),
        ("EVALUATION_RAGAS_MODEL", " ", "EVALUATION_RAGAS_MODEL must not be blank"),
        (
            "EVALUATION_RAGAS_EMBEDDING_MODEL",
            " ",
            "EVALUATION_RAGAS_EMBEDDING_MODEL must not be blank",
        ),
        ("EVALUATION_RAGAS_TIMEOUT", "0", "EVALUATION_RAGAS_TIMEOUT must be positive"),
        ("EVALUATION_RAGAS_TIMEOUT", "nan", "EVALUATION_RAGAS_TIMEOUT must be positive"),
        (
            "EVALUATION_FAITHFULNESS_THRESHOLD",
            "1.1",
            "EVALUATION_FAITHFULNESS_THRESHOLD must be between 0 and 1",
        ),
        (
            "EVALUATION_CONTEXT_RECALL_THRESHOLD",
            "-0.1",
            "EVALUATION_CONTEXT_RECALL_THRESHOLD must be between 0 and 1",
        ),
        (
            "EVALUATION_QUESTION_PREVIEW_CHARS",
            "0",
            "EVALUATION_QUESTION_PREVIEW_CHARS must be a positive integer",
        ),
        (
            "EVALUATION_ANSWER_PREVIEW_CHARS",
            "nope",
            "EVALUATION_ANSWER_PREVIEW_CHARS must be a positive integer",
        ),
    ],
)
def test_evaluation_config_rejects_invalid_values(monkeypatch, name, value, message):
    with pytest.raises(RuntimeError, match=message):
        reload_config(monkeypatch, **{name: value})


def test_faq_config_reads_environment_overrides(monkeypatch, tmp_path):
    config = reload_config(
        monkeypatch,
        FAQ_ENABLED="false",
        FAQ_DB_PATH=str(tmp_path / "custom.db"),
        FAQ_MATCH_THRESHOLD="2.5",
        FAQ_MATCH_MARGIN="0.4",
        FAQ_CACHE_ENABLED="false",
        FAQ_CACHE_TTL_SECONDS="60",
        FAQ_CACHE_PREWARM="false",
        REDIS_URL="redis://cache:6380/2",
        REDIS_CONNECT_TIMEOUT_SECONDS="0.5",
        REDIS_SOCKET_TIMEOUT_SECONDS="0.6",
    )

    assert config.FAQ_ENABLED is False
    assert config.FAQ_DB_PATH == tmp_path / "custom.db"
    assert config.FAQ_MATCH_THRESHOLD == 2.5
    assert config.FAQ_MATCH_MARGIN == 0.4
    assert config.FAQ_CACHE_ENABLED is False
    assert config.FAQ_CACHE_TTL_SECONDS == 60
    assert config.FAQ_CACHE_PREWARM is False
    assert config.REDIS_URL == "redis://cache:6380/2"
    assert config.REDIS_CONNECT_TIMEOUT_SECONDS == 0.5
    assert config.REDIS_SOCKET_TIMEOUT_SECONDS == 0.6


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("FAQ_MATCH_THRESHOLD", "-1", "FAQ_MATCH_THRESHOLD"),
        ("FAQ_MATCH_MARGIN", "-0.1", "FAQ_MATCH_MARGIN"),
        ("FAQ_CACHE_TTL_SECONDS", "0", "FAQ_CACHE_TTL_SECONDS"),
        ("REDIS_CONNECT_TIMEOUT_SECONDS", "0", "REDIS_CONNECT_TIMEOUT_SECONDS"),
        ("REDIS_SOCKET_TIMEOUT_SECONDS", "-1", "REDIS_SOCKET_TIMEOUT_SECONDS"),
        ("FAQ_ENABLED", "perhaps", "FAQ_ENABLED must be a boolean"),
        ("FAQ_CACHE_PREWARM", "perhaps", "FAQ_CACHE_PREWARM must be a boolean"),
        ("FAQ_DB_PATH", " ", "FAQ_DB_PATH must not be blank"),
        ("REDIS_URL", " ", "REDIS_URL must not be blank"),
    ],
)
def test_faq_config_rejects_invalid_values(monkeypatch, name, value, message):
    with pytest.raises(RuntimeError, match=message):
        reload_config(monkeypatch, **{name: value})


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

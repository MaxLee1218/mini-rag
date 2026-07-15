from __future__ import annotations

import math
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "rag_requests.jsonl"

if os.getenv("MINI_RAG_SKIP_DOTENV") != "1":
    load_dotenv(ENV_PATH)


def _parse_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return float(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a valid number") from error


def _parse_nonnegative_float_env(name: str, default: float) -> float:
    parsed = _parse_float_env(name, default)
    if parsed < 0:
        raise RuntimeError(f"{name} must be non-negative")
    return parsed


def _parse_positive_float_env(name: str, default: float) -> float:
    parsed = _parse_float_env(name, default)
    if parsed <= 0:
        raise RuntimeError(f"{name} must be positive")
    return parsed


def _parse_finite_positive_float_env(name: str, default: float) -> float:
    parsed = _parse_float_env(name, default)
    if not math.isfinite(parsed) or parsed <= 0:
        raise RuntimeError(f"{name} must be positive")
    return parsed


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean")


def _parse_positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return parsed


def _parse_nonnegative_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a non-negative integer") from error
    if parsed < 0:
        raise RuntimeError(f"{name} must be a non-negative integer")
    return parsed


def _parse_bounded_int_env(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(
            f"{name} must be an integer between {minimum} and {maximum}"
        ) from error
    if not minimum <= parsed <= maximum:
        raise RuntimeError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return parsed


def _parse_choice_env(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        raise RuntimeError(f"{name} must be one of: {', '.join(choices)}")
    return value


def _parse_nonblank_env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise RuntimeError(f"{name} must not be blank")
    return value


def _parse_project_path_env(name: str, default: str) -> Path:
    path = Path(_parse_nonblank_env(name, default))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _parse_unit_interval_env(name: str, default: float) -> float:
    parsed = _parse_float_env(name, default)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise RuntimeError(f"{name} must be between 0 and 1")
    return parsed


VECTOR_DB_PATH = "data/chroma"
VECTOR_COLLECTION_NAME = "mini_rag_chunks"
DEFAULT_TOP_K = 5
CHUNK_MODE = _parse_choice_env(
    "RAG_CHUNK_MODE", "standard", ("standard", "parent-child")
)
PARENT_CHUNK_SIZE = _parse_positive_int_env("RAG_PARENT_CHUNK_SIZE", 1000)
PARENT_CHUNK_OVERLAP = _parse_nonnegative_int_env(
    "RAG_PARENT_CHUNK_OVERLAP", 100
)
CHILD_CHUNK_SIZE = _parse_positive_int_env("RAG_CHILD_CHUNK_SIZE", 250)
CHILD_CHUNK_OVERLAP = _parse_nonnegative_int_env("RAG_CHILD_CHUNK_OVERLAP", 50)
PARENT_STORE_PATH = _parse_nonblank_env(
    "RAG_PARENT_STORE_PATH", "data/parents/parents.sqlite3"
)
if CHILD_CHUNK_SIZE > PARENT_CHUNK_SIZE:
    raise RuntimeError(
        "RAG_CHILD_CHUNK_SIZE must be smaller than or equal to RAG_PARENT_CHUNK_SIZE"
    )
if PARENT_CHUNK_OVERLAP >= PARENT_CHUNK_SIZE:
    raise RuntimeError(
        "RAG_PARENT_CHUNK_OVERLAP must be smaller than RAG_PARENT_CHUNK_SIZE"
    )
if CHILD_CHUNK_OVERLAP >= CHILD_CHUNK_SIZE:
    raise RuntimeError(
        "RAG_CHILD_CHUNK_OVERLAP must be smaller than RAG_CHILD_CHUNK_SIZE"
    )
HYBRID_SPARSE_WEIGHT = 0.5
HYBRID_DENSE_WEIGHT = 0.5
HYBRID_TOP_K = 5
HYBRID_CANDIDATE_MULTIPLIER = 2

FAQ_ENABLED = _parse_bool_env("FAQ_ENABLED", True)
_FAQ_DB_PATH_VALUE = _parse_nonblank_env("FAQ_DB_PATH", "data/faq.db")
FAQ_DB_PATH = Path(_FAQ_DB_PATH_VALUE)
if not FAQ_DB_PATH.is_absolute():
    FAQ_DB_PATH = PROJECT_ROOT / FAQ_DB_PATH
FAQ_MATCH_THRESHOLD = _parse_nonnegative_float_env("FAQ_MATCH_THRESHOLD", 1.0)
FAQ_MATCH_MARGIN = _parse_nonnegative_float_env("FAQ_MATCH_MARGIN", 0.15)
FAQ_CACHE_ENABLED = _parse_bool_env("FAQ_CACHE_ENABLED", True)
FAQ_CACHE_TTL_SECONDS = _parse_positive_int_env("FAQ_CACHE_TTL_SECONDS", 86400)
FAQ_CACHE_PREWARM = _parse_bool_env("FAQ_CACHE_PREWARM", True)
REDIS_URL = _parse_nonblank_env("REDIS_URL", "redis://localhost:6379/0")
REDIS_CONNECT_TIMEOUT_SECONDS = _parse_positive_float_env(
    "REDIS_CONNECT_TIMEOUT_SECONDS", 0.2
)
REDIS_SOCKET_TIMEOUT_SECONDS = _parse_positive_float_env(
    "REDIS_SOCKET_TIMEOUT_SECONDS", 0.2
)

CONVERSATION_HISTORY_LIMIT = _parse_bounded_int_env(
    "CONVERSATION_HISTORY_LIMIT", 5, 3, 5
)
QUERY_REWRITE_ENABLED = _parse_bool_env("QUERY_REWRITE_ENABLED", True)
QUERY_REWRITE_PROVIDER = _parse_choice_env(
    "QUERY_REWRITE_PROVIDER", "deepseek", ("deepseek",)
)
QUERY_REWRITE_TIMEOUT = _parse_float_env("QUERY_REWRITE_TIMEOUT", 10.0)
if QUERY_REWRITE_TIMEOUT <= 0:
    raise RuntimeError("QUERY_REWRITE_TIMEOUT must be positive")

RERANKER_ENABLED = _parse_bool_env("RERANKER_ENABLED", True)
RERANKER_MODEL = _parse_nonblank_env(
    "RERANKER_MODEL", "cross-encoder/ms-marco-TinyBERT-L2-v2"
)
RERANKER_TOP_K = _parse_positive_int_env("RERANKER_TOP_K", 5)
RERANKER_CANDIDATE_K = _parse_positive_int_env("RERANKER_CANDIDATE_K", 10)
RERANKER_BATCH_SIZE = _parse_positive_int_env("RERANKER_BATCH_SIZE", 16)
RERANKER_MAX_LENGTH = _parse_positive_int_env("RERANKER_MAX_LENGTH", 256)
RERANKER_DEVICE = _parse_choice_env(
    "RERANKER_DEVICE", "cpu", ("auto", "cpu", "cuda", "mps")
)
RERANKER_FAILURE_MODE = _parse_choice_env(
    "RERANKER_FAILURE_MODE", "fallback", ("fallback",)
)
RERANKER_LOCAL_FILES_ONLY = _parse_bool_env("RERANKER_LOCAL_FILES_ONLY", False)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_TIMEOUT = _parse_float_env("DEEPSEEK_TIMEOUT", 30.0)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

EVALUATION_DATASET_PATH = _parse_project_path_env(
    "EVALUATION_DATASET_PATH", "evaluation/dataset/eval_dataset.json"
)
EVALUATION_JSON_REPORT_PATH = _parse_project_path_env(
    "EVALUATION_JSON_REPORT_PATH", "reports/evaluation_report.json"
)
EVALUATION_MARKDOWN_REPORT_PATH = _parse_project_path_env(
    "EVALUATION_MARKDOWN_REPORT_PATH", "reports/evaluation_report.md"
)
EVALUATION_TOP_K = _parse_positive_int_env("EVALUATION_TOP_K", 5)
EVALUATION_RAGAS_MODEL = _parse_nonblank_env(
    "EVALUATION_RAGAS_MODEL", "gpt-4o-mini"
)
EVALUATION_RAGAS_EMBEDDING_MODEL = _parse_nonblank_env(
    "EVALUATION_RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"
)
EVALUATION_RAGAS_TIMEOUT = _parse_finite_positive_float_env(
    "EVALUATION_RAGAS_TIMEOUT", 60.0
)
EVALUATION_FAITHFULNESS_THRESHOLD = _parse_unit_interval_env(
    "EVALUATION_FAITHFULNESS_THRESHOLD", 0.7
)
EVALUATION_CONTEXT_RECALL_THRESHOLD = _parse_unit_interval_env(
    "EVALUATION_CONTEXT_RECALL_THRESHOLD", 0.7
)
EVALUATION_QUESTION_PREVIEW_CHARS = _parse_positive_int_env(
    "EVALUATION_QUESTION_PREVIEW_CHARS", 300
)
EVALUATION_ANSWER_PREVIEW_CHARS = _parse_positive_int_env(
    "EVALUATION_ANSWER_PREVIEW_CHARS", 800
)


def require_deepseek_api_key() -> str:
    """Return the DeepSeek API key or raise a clear setup error."""
    if not isinstance(DEEPSEEK_API_KEY, str) or not DEEPSEEK_API_KEY.strip():
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Please create a .env file in the project root "
            "based on .env.example and set your DeepSeek API key."
        )
    return DEEPSEEK_API_KEY.strip()


def require_openai_api_key() -> str:
    """Return the OpenAI API key or raise a clear setup error."""
    if not isinstance(OPENAI_API_KEY, str) or not OPENAI_API_KEY.strip():
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Please create a .env file in the project root "
            "based on .env.example and set your OpenAI API key."
        )
    return OPENAI_API_KEY.strip()

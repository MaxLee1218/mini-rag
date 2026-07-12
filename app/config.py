from __future__ import annotations

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


VECTOR_DB_PATH = "data/chroma"
VECTOR_COLLECTION_NAME = "mini_rag_chunks"
DEFAULT_TOP_K = 5
HYBRID_SPARSE_WEIGHT = 0.5
HYBRID_DENSE_WEIGHT = 0.5
HYBRID_TOP_K = 5
HYBRID_CANDIDATE_MULTIPLIER = 2

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

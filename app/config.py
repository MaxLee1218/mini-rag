from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

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


VECTOR_DB_PATH = "data/chroma"
VECTOR_COLLECTION_NAME = "mini_rag_chunks"
DEFAULT_TOP_K = 4

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


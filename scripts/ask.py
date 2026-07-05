from __future__ import annotations

import argparse
import inspect
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import VECTOR_COLLECTION_NAME, VECTOR_DB_PATH
from app.embeddings import Embedder
from app.generator import (
    DeepSeekGenerator,
    MissingAPIKeyError,
    load_deepseek_config_from_env,
)
from app.pipeline import RAGPipeline

try:
    from app.prompt_builder import extract_context_text as _prompt_extract_context_text
except ImportError:
    _prompt_extract_context_text = None

from app.retriever import Retriever
from app.vector_store import ChromaVectorStore


DEFAULT_TOP_K = 4
CONTEXT_PREVIEW_CHARS = 300
EXIT_COMMANDS = {"exit", "quit", "q"}
SOURCE_FIELDS = ("source", "file_path", "filename", "path", "title")
SOURCE_MARKERS = ("来源：", "来源:", "Sources:", "Source:", "sources:", "source:")


class AskCLIError(Exception):
    """Base exception for expected CLI errors."""


class EmptyQuestionError(AskCLIError):
    """Raised when the user provides a blank question."""


class VectorStoreNotReadyError(AskCLIError):
    """Raised when the local vector database is missing or empty."""


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--top-k must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--top-k must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions against the local mini-rag knowledge base.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask. Omit this argument to enter interactive mode.",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved chunks to use.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print short debug previews of retrieved contexts.",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Do not print an extra sources section from result.sources.",
    )
    return parser.parse_args(argv)


def build_default_pipeline(top_k: int) -> RAGPipeline:
    _validate_top_k(top_k)
    config = load_deepseek_config_from_env()
    persist_path = _resolved_vector_db_path()
    if not persist_path.exists():
        raise VectorStoreNotReadyError("请先运行 scripts/ingest.py")

    embedder = Embedder()
    vector_store = ChromaVectorStore(
        collection_name=VECTOR_COLLECTION_NAME,
        persist_path=str(persist_path),
    )
    try:
        count = _get_vector_store_count(vector_store)
        if count == 0:
            raise VectorStoreNotReadyError("请先运行 scripts/ingest.py")

        retriever_kwargs = {
            "embedder": embedder,
            "vector_store": vector_store,
        }
        if not _pipeline_type_supports_top_k(RAGPipeline):
            retriever_kwargs["default_top_k"] = top_k

        retriever = Retriever(**retriever_kwargs)
        generator = DeepSeekGenerator(config=config)
        return RAGPipeline(retriever=retriever, generator=generator)
    except Exception:
        close = getattr(vector_store, "close", None)
        if callable(close):
            close()
        raise


def format_result(
    result: Any,
    show_context: bool = False,
    show_sources: bool = True,
) -> str:
    question = _display_value(getattr(result, "question", "")) or ""
    answer = _display_value(getattr(result, "answer", "")) or ""
    contexts = _as_list(getattr(result, "contexts", []))
    sources = _dedupe_sources(getattr(result, "sources", []))

    sections = [
        "问题：",
        question,
        "",
        "回答：",
        answer,
    ]

    if show_context:
        sections.extend(["", _format_contexts(contexts)])

    if show_sources and sources and not _answer_has_sources(answer):
        sections.extend(["", "来源："])
        sections.extend(
            f"[{index}] {source}" for index, source in enumerate(sources, 1)
        )

    return "\n".join(sections).rstrip()


def ask_once(
    question: str,
    pipeline: Any,
    show_context: bool = False,
    show_sources: bool = True,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    clean_question = _clean_question(question)
    _validate_top_k(top_k)
    result = _call_pipeline(pipeline, clean_question, top_k=top_k)
    contexts = getattr(result, "contexts", None)
    if contexts is not None and not list(contexts):
        raise VectorStoreNotReadyError("请先运行 scripts/ingest.py")
    return format_result(
        result,
        show_context=show_context,
        show_sources=show_sources,
    )


def interactive_loop(
    pipeline: Any,
    show_context: bool = False,
    show_sources: bool = True,
    top_k: int = DEFAULT_TOP_K,
) -> int:
    print("请输入问题，输入 exit / quit / q 退出")
    while True:
        try:
            question = input("> ")
        except EOFError:
            print()
            return 0

        clean_question = question.strip()
        if not clean_question:
            continue
        if clean_question.lower() in EXIT_COMMANDS:
            return 0

        print(
            ask_once(
                clean_question,
                pipeline,
                show_context=show_context,
                show_sources=show_sources,
                top_k=top_k,
            )
        )
        print()


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as error:
        return _system_exit_code(error)

    pipeline = None
    try:
        if args.question is not None and not args.question.strip():
            raise EmptyQuestionError("问题不能为空")

        pipeline = build_default_pipeline(args.top_k)
        if args.question is None:
            return interactive_loop(
                pipeline,
                show_context=args.show_context,
                show_sources=not args.no_sources,
                top_k=args.top_k,
            )

        print(
            ask_once(
                args.question,
                pipeline,
                show_context=args.show_context,
                show_sources=not args.no_sources,
                top_k=args.top_k,
            )
        )
        return 0
    except EmptyQuestionError:
        print("错误：问题不能为空", file=sys.stderr)
        return 1
    except MissingAPIKeyError:
        print("错误：请检查 .env 中的 DEEPSEEK_API_KEY", file=sys.stderr)
        return 1
    except VectorStoreNotReadyError:
        print("错误：请先运行 scripts/ingest.py", file=sys.stderr)
        return 1
    except ValueError as error:
        error_message = str(error)
        if (
            "question must not be blank" in error_message
            or "query must not be blank" in error_message
        ):
            print("错误：问题不能为空", file=sys.stderr)
        else:
            print(f"错误：{error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    finally:
        if pipeline is not None:
            close_pipeline_resources(pipeline)


def close_pipeline_resources(pipeline: Any) -> None:
    try:
        vector_store = getattr(
            getattr(pipeline, "retriever", None),
            "vector_store",
            None,
        )
        close = getattr(vector_store, "close", None)
        if callable(close):
            close()
    except Exception as error:
        print(f"警告：关闭向量库失败：{error}", file=sys.stderr)


def _call_pipeline(pipeline: Any, question: str, *, top_k: int) -> Any:
    for method_name in ("ask", "run", "query"):
        method = getattr(pipeline, method_name, None)
        if not callable(method):
            continue
        if _callable_supports_top_k(method):
            return method(question, top_k=top_k)
        return method(question)
    raise RuntimeError("RAG pipeline must provide ask(), run(), or query()")


def _get_vector_store_count(vector_store: Any) -> int | None:
    count = getattr(vector_store, "count", None)
    if callable(count):
        try:
            return int(count())
        except Exception:
            return None

    collection = getattr(vector_store, "collection", None)
    collection_count = getattr(collection, "count", None)
    if callable(collection_count):
        try:
            return int(collection_count())
        except Exception:
            return None

    return None


def _pipeline_type_supports_top_k(pipeline_type: Any) -> bool:
    for method_name in ("ask", "run", "query"):
        method = getattr(pipeline_type, method_name, None)
        if callable(method):
            return _callable_supports_top_k(method)
    return False


def _callable_supports_top_k(callable_object: Any) -> bool:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "top_k" in signature.parameters


def _format_contexts(contexts: list[Any]) -> str:
    if not contexts:
        return "检索上下文：\n(no context)"

    lines = ["检索上下文："]
    for index, context in enumerate(contexts, start=1):
        source = _context_source(context)
        text = extract_context_text(context)
        preview = _preview_text(text, CONTEXT_PREVIEW_CHARS) if text else "(empty)"
        lines.extend([f"[{index}] {source}", preview])
    return "\n".join(lines)


def extract_context_text(context: Any) -> str:
    if _prompt_extract_context_text is not None:
        text = _prompt_extract_context_text(context)
        if text:
            return text
    return _fallback_extract_context_text(context)


def _fallback_extract_context_text(context: Any) -> str:
    text_fields = ("text", "document", "content", "page_content", "chunk")
    if isinstance(context, str):
        return context.strip()
    if isinstance(context, Mapping):
        for field_name in text_fields:
            text = _display_value(context.get(field_name))
            if text:
                return text
        return ""

    for field_name in text_fields:
        text = _display_value(getattr(context, field_name, None))
        if text:
            return text
    return ""


def _context_source(context: Any) -> str:
    if isinstance(context, Mapping):
        for field_name in SOURCE_FIELDS:
            value = _display_value(context.get(field_name))
            if value:
                return value

        metadata = context.get("metadata")
        if isinstance(metadata, Mapping):
            for field_name in SOURCE_FIELDS:
                value = _display_value(metadata.get(field_name))
                if value:
                    return value
        return "unknown source"

    for field_name in SOURCE_FIELDS:
        value = _display_value(getattr(context, field_name, None))
        if value:
            return value

    metadata = getattr(context, "metadata", None)
    if isinstance(metadata, Mapping):
        for field_name in SOURCE_FIELDS:
            value = _display_value(metadata.get(field_name))
            if value:
                return value
    return "unknown source"


def _dedupe_sources(sources: Any) -> list[str]:
    if sources is None:
        return []
    if isinstance(sources, (str, bytes)):
        source_items = [sources]
    else:
        try:
            source_items = list(sources)
        except TypeError:
            source_items = [sources]

    deduped = []
    seen = set()
    for source in source_items:
        formatted = _format_source(source)
        if not formatted or formatted in seen:
            continue
        seen.add(formatted)
        deduped.append(formatted)
    return deduped


def _format_source(source: Any) -> str:
    if isinstance(source, Mapping):
        for field_name in SOURCE_FIELDS:
            value = _display_value(source.get(field_name))
            if value:
                return value
        return ""
    return _display_value(source) or ""


def _answer_has_sources(answer: str) -> bool:
    return any(marker in answer for marker in SOURCE_MARKERS)


def _preview_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _clean_question(question: Any) -> str:
    if not isinstance(question, str) or not question.strip():
        raise EmptyQuestionError("问题不能为空")
    return question.strip()


def _validate_top_k(top_k: Any) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return top_k


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _display_value(value: Any) -> str | None:
    if value is None or callable(value):
        return None
    if isinstance(value, str):
        clean_value = value.strip()
        return clean_value or None
    return str(value)


def _resolved_vector_db_path() -> Path:
    path = Path(VECTOR_DB_PATH)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _system_exit_code(error: SystemExit) -> int:
    code = error.code
    if isinstance(code, int):
        return code
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

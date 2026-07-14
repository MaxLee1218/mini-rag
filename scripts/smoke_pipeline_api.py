from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_TOP_K
from app.pipeline import RAGResult
from app.pipeline_factory import get_default_dual_path_pipeline
from app.prompt_builder import extract_context_text


DEFAULT_QUESTION = "我今年几岁？"
CONTEXT_PREVIEW_CHARS = 500
PROMPT_PREVIEW_CHARS = 3000
SOURCE_DISPLAY_FIELDS = ("source", "title", "path", "file_path", "filename")


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--top-k must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--top-k must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real RAGPipeline smoke test against the local vector store "
            "and DeepSeek API."
        )
    )
    parser.add_argument(
        "positional_question",
        nargs="?",
        default=None,
        help="Question to ask the RAG pipeline.",
    )
    parser.add_argument(
        "--question",
        dest="question_option",
        help="Question to ask the selected route.",
    )
    parser.add_argument(
        "--route",
        choices=("faq", "rag"),
        default="rag",
        help="Route being exercised by this smoke run.",
    )
    parser.add_argument(
        "--expect-route",
        choices=("faq", "rag"),
        help="Exit nonzero unless the result uses this route.",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved contexts to use.",
    )
    parser.add_argument(
        "--debug-context",
        action="store_true",
        help="Print retrieved context metadata and text previews.",
    )
    parser.add_argument(
        "--debug-prompt",
        action="store_true",
        help="Print the prompt preview sent to the generator.",
    )
    args = parser.parse_args(argv)
    args.question = args.question_option or args.positional_question or DEFAULT_QUESTION
    return args


def build_pipeline() -> Any:
    return get_default_dual_path_pipeline()


def print_result(result: RAGResult, *, latency_ms: float) -> None:
    print("=== RAG Pipeline Smoke Test ===")
    print()
    print(f"route: {result.route}")
    print(f"latency_ms: {latency_ms:.3f}")
    if result.route == "faq":
        print(f"faq_id: {result.faq_id}")
        print(f"faq_score: {result.faq_score}")
        print(f"faq_match_type: {result.faq_match_type}")
        print(f"faq_cache_hit: {str(result.faq_cache_hit).lower()}")
    print()
    print("Question:")
    print(result.question)
    print()
    print("Answer:")
    print(result.answer)
    print()
    print("Sources:")
    if result.sources:
        for source in result.sources:
            print(f"- {_format_source(source)}")
    else:
        print("(none)")
    print()
    print("Context count:")
    print(len(result.contexts))
    print()

    if result.route == "rag" and not result.contexts:
        print(
            "WARNING: Retriever returned 0 contexts. Please check whether data/chroma "
            "exists, ingest has run, collection name is correct, and the query is "
            "related to the documents."
        )
    if result.route == "rag" and not result.sources:
        print(
            "WARNING: No sources were returned. Please check whether metadata.source "
            "is preserved during ingest/chunk/retrieval."
        )

    print()
    print("Status:")
    print("OK")


def print_debug_contexts(result: RAGResult) -> None:
    print("=== Debug Contexts ===")
    print()
    if not result.contexts:
        print("(no contexts)")
        print()
        return

    for index, context in enumerate(result.contexts, start=1):
        text = extract_context_text(context)
        print(f"[{index}]")
        print(f"Type: {type(context).__name__}")
        print(f"Source: {_context_display_value(context, SOURCE_DISPLAY_FIELDS)}")
        print(f"Score: {_context_display_value(context, ('score',))}")
        print(f"Distance: {_context_display_value(context, ('distance',))}")
        print(f'Contains "max is my name": {"max is my name" in text}')
        print(f'Contains "114514": {"114514" in text}')
        print("Preview:")
        print(_preview_text(text, CONTEXT_PREVIEW_CHARS) if text else "(empty)")
        print()


def print_debug_prompt(result: RAGResult) -> None:
    print("=== Debug Prompt ===")
    print()
    prompt = result.prompt or ""
    if not prompt:
        print("(prompt unavailable)")
        print()
        return

    print(f'Prompt contains "max is my name": {"max is my name" in prompt}')
    print(f'Prompt contains "114514": {"114514" in prompt}')
    print(f"Prompt contains question: {result.question in prompt}")
    print()
    print("Prompt preview:")
    print(_preview_text(prompt, PROMPT_PREVIEW_CHARS))
    print()


def close_pipeline_resources(pipeline: Any) -> None:
    try:
        retriever = getattr(pipeline, "retriever", None)
        close_retriever = getattr(retriever, "close", None)
        if callable(close_retriever):
            close_retriever()
            return
        dense_retriever = getattr(retriever, "dense_retriever", retriever)
        vector_store = getattr(dense_retriever, "vector_store", None)
        close = getattr(vector_store, "close", None)
        if callable(close):
            close()
    except Exception as close_error:
        print(
            f"WARNING: Failed to close vector store cleanly: {close_error}",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline: Any | None = None
    try:
        pipeline = build_pipeline()
        started_at = time.perf_counter()
        result = pipeline.ask(args.question, top_k=args.top_k)
        latency_ms = (time.perf_counter() - started_at) * 1000
        if args.debug_context:
            print_debug_contexts(result)
        if args.debug_prompt:
            print_debug_prompt(result)
        print_result(result, latency_ms=latency_ms)
        expected_route = args.expect_route or args.route
        if result.route != expected_route:
            print(
                f"ERROR: expected route {expected_route}, got {result.route}",
                file=sys.stderr,
            )
            return 1
    except Exception as exc:
        print(f"ERROR: Smoke test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if pipeline is not None:
            close_pipeline_resources(pipeline)
    return 0


def _format_source(source: Any) -> str:
    if isinstance(source, Mapping):
        for field_name in SOURCE_DISPLAY_FIELDS:
            value = source.get(field_name)
            if value is not None and str(value).strip():
                return str(value).strip()
    return str(source)


def _context_display_value(context: Any, field_names: tuple[str, ...]) -> str:
    if isinstance(context, Mapping):
        metadata = context.get("metadata")
        for field_name in field_names:
            value = context.get(field_name)
            if value is not None and str(value).strip():
                return str(value).strip()
        if isinstance(metadata, Mapping):
            for field_name in field_names:
                value = metadata.get(field_name)
                if value is not None and str(value).strip():
                    return str(value).strip()
        return "unknown"

    metadata = getattr(context, "metadata", None)
    for field_name in field_names:
        value = getattr(context, field_name, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    if isinstance(metadata, Mapping):
        for field_name in field_names:
            value = metadata.get(field_name)
            if value is not None and str(value).strip():
                return str(value).strip()
    return "unknown"


def _preview_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


if __name__ == "__main__":
    raise SystemExit(main())

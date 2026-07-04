from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import VECTOR_COLLECTION_NAME, VECTOR_DB_PATH
from app.embeddings import Embedder
from app.generator import DeepSeekGenerator, load_deepseek_config_from_env
from app.pipeline import RAGPipeline, RAGResult
from app.prompt_builder import extract_context_text
from app.retriever import Retriever
from app.vector_store import ChromaVectorStore


DEFAULT_QUESTION = "我今年几岁？"
DEFAULT_TOP_K = 4
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
        description="Run a real RAGPipeline smoke test against the local vector store and DeepSeek API."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Question to ask the RAG pipeline.",
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
    return parser.parse_args(argv)


def build_pipeline() -> RAGPipeline:
    config = load_deepseek_config_from_env()
    embedder = Embedder()
    vector_store = ChromaVectorStore(
        collection_name=VECTOR_COLLECTION_NAME,
        persist_path=VECTOR_DB_PATH,
    )
    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    generator = DeepSeekGenerator(config=config)
    return RAGPipeline(retriever=retriever, generator=generator)


def print_result(result: RAGResult) -> None:
    print("=== RAG Pipeline Smoke Test ===")
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

    if not result.contexts:
        print(
            "WARNING: Retriever returned 0 contexts. Please check whether data/chroma "
            "exists, ingest has run, collection name is correct, and the query is "
            "related to the documents."
        )
    if not result.sources:
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


def close_pipeline_resources(pipeline: RAGPipeline) -> None:
    try:
        vector_store = getattr(getattr(pipeline, "retriever", None), "vector_store", None)
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
    pipeline: RAGPipeline | None = None
    try:
        pipeline = build_pipeline()
        result = pipeline.ask(args.question, top_k=args.top_k)
        if args.debug_context:
            print_debug_contexts(result)
        if args.debug_prompt:
            print_debug_prompt(result)
        print_result(result)
    except Exception as exc:
        print(f"ERROR: Smoke test failed: {exc}", file=sys.stderr)
        raise
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

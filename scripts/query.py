from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    CHUNK_MODE,
    PARENT_STORE_PATH,
    VECTOR_COLLECTION_NAME,
    VECTOR_DB_PATH,
)
from app.embeddings import Embedder
from app.parent_store import SQLiteParentStore
from app.retriever import Retriever
from app.vector_store import ChromaVectorStore


DEFAULT_TOP_K = 3
DEFAULT_MAX_CHARS = 1000
UNKNOWN = "unknown"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve source-attributed evidence from a local RAG collection."
    )
    parser.add_argument("--query", help="Question to retrieve evidence for.")
    parser.add_argument(
        "--collection",
        default=VECTOR_COLLECTION_NAME,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--persist-path",
        default=VECTOR_DB_PATH,
        help="Chroma persistent database path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--chunk-mode",
        choices=("standard", "parent-child"),
        default=CHUNK_MODE,
        help="Retrieval mode. In parent-child mode top-k counts child hits.",
    )
    parser.add_argument(
        "--parent-store-path",
        default=PARENT_STORE_PATH,
        help="SQLite parent store used in parent-child mode.",
    )
    parser.add_argument(
        "--where",
        help='Metadata filter JSON, for example: \'{ "filename": "rag_notes.txt" }\'',
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Also print Retriever.build_context(results).",
    )
    parser.add_argument(
        "--show-child",
        action="store_true",
        help="Show the highest-ranked matched child for each restored parent.",
    )
    parser.add_argument(
        "--show-parent-id",
        action="store_true",
        help="Show internal parent IDs for diagnostics.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum displayed characters per retrieved chunk.",
    )
    parser.add_argument(
        "--no-sources-summary",
        action="store_true",
        help="Skip the final unique Sources section.",
    )
    return parser.parse_args(argv)


def parse_where_filter(where_text: str | None) -> dict[str, Any] | None:
    if where_text is None:
        return None

    try:
        parsed = json.loads(where_text)
    except json.JSONDecodeError as error:
        parsed = _parse_simple_where_object(where_text, error)

    if not isinstance(parsed, dict):
        raise ValueError("--where must be a JSON object")
    return parsed


def truncate_text(text: str, max_chars: int) -> str:
    _validate_positive_int(max_chars, "max_chars")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def extract_source_info(result: Mapping[str, Any]) -> dict[str, str]:
    metadata = result.get("metadata") if isinstance(result, Mapping) else {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    return {
        "source": _clean_display_value(metadata.get("source")),
        "filename": _clean_display_value(metadata.get("filename")),
        "relative_path": _clean_display_value(metadata.get("relative_path")),
    }


def format_result(
    result: Mapping[str, Any],
    index: int,
    max_chars: int = DEFAULT_MAX_CHARS,
    *,
    show_child: bool = False,
    show_parent_id: bool = False,
) -> str:
    source_info = extract_source_info(result)
    chunk_id = _clean_display_value(result.get("id"))
    distance = _format_distance(result.get("distance"))
    text = _result_text(result)
    displayed_text = truncate_text(text, max_chars)

    lines = [
            f"[{index}] Source: {source_info['source']}",
            f"Filename: {source_info['filename']}",
            f"Relative path: {source_info['relative_path']}",
            f"Chunk ID: {chunk_id}",
            f"Distance: {distance}",
            "Text:",
            displayed_text,
        ]
    if show_parent_id:
        lines.insert(4, f"Parent ID: {chunk_id}")
    if show_child:
        retrieval = result.get("retrieval")
        if isinstance(retrieval, Mapping):
            lines.extend(
                [
                    f"Matched child ID: {_clean_display_value(retrieval.get('matched_child_id'))}",
                    f"Child score: {_format_distance(retrieval.get('child_score'))}",
                    "Matched child text:",
                    truncate_text(
                        str(retrieval.get("matched_child_text") or ""), max_chars
                    ),
                ]
            )
    return "\n".join(lines)


def format_sources(results: Sequence[Mapping[str, Any]]) -> str:
    sources = []
    seen = set()

    for result in results:
        source = _display_source(result)
        if source in seen:
            continue
        seen.add(source)
        sources.append(source)

    return "Sources:\n" + "\n".join(f"- {source}" for source in sources)


def format_no_results() -> str:
    return "\n".join(
        [
            "No retrieved results returned.",
            "",
            "Hints:",
            "- Make sure you have run scripts/ingest.py first.",
            "- Make sure you are querying the same collection name used during ingestion.",
            "- Try increasing --top-k.",
        ]
    )


def run_query(
    query: Any,
    collection: Any,
    persist_path: Any,
    top_k: Any,
    where: Mapping[str, Any] | None = None,
    show_context: bool = False,
    max_chars: Any = DEFAULT_MAX_CHARS,
    include_sources_summary: bool = True,
    chunk_mode: str = CHUNK_MODE,
    parent_store_path: str | Path = PARENT_STORE_PATH,
    show_child: bool = False,
    show_parent_id: bool = False,
    retriever: Any | None = None,
) -> str:
    clean_query = _validate_nonblank_string(query, "query")
    clean_collection = _validate_nonblank_string(collection, "collection")
    clean_persist_path = _validate_nonblank_string(persist_path, "persist_path")
    resolved_top_k = _validate_positive_int(top_k, "top_k")
    resolved_max_chars = _validate_positive_int(max_chars, "max_chars")
    if chunk_mode not in {"standard", "parent-child"}:
        raise ValueError("chunk_mode must be one of: standard, parent-child")

    active_retriever = retriever
    created_store = None
    created_parent_store = None
    try:
        if active_retriever is None:
            if chunk_mode == "parent-child":
                resolved_parent_store_path = _resolved_project_path(parent_store_path)
                if not resolved_parent_store_path.is_file():
                    raise ValueError(
                        "parent store is missing; run parent-child ingest first: "
                        f"{resolved_parent_store_path}"
                    )
                created_parent_store = SQLiteParentStore(
                    resolved_parent_store_path
                )
            embedder = Embedder()
            created_store = ChromaVectorStore(
                collection_name=clean_collection,
                persist_path=clean_persist_path,
            )
            active_retriever = Retriever(
                embedder=embedder,
                vector_store=created_store,
                default_top_k=resolved_top_k,
                mode=chunk_mode,
                parent_store=created_parent_store,
            )

        results = active_retriever.retrieve(
            clean_query,
            top_k=resolved_top_k,
            where=where,
        )
        if results is None:
            results = []
        results = list(results)

        sections = [
            f"Query: {clean_query}",
            f"Collection: {clean_collection}",
            f"Top k: {resolved_top_k}",
            "",
            f"Retrieved results: {len(results)}",
        ]

        if not results:
            sections.extend(["", format_no_results()])
            return "\n".join(sections)

        for index, result in enumerate(results, start=1):
            sections.extend(
                [
                    "",
                    format_result(
                        result,
                        index=index,
                        max_chars=resolved_max_chars,
                        show_child=show_child,
                        show_parent_id=show_parent_id,
                    ),
                ]
            )

        if show_context:
            sections.extend(["", _format_context(active_retriever, results)])

        if include_sources_summary:
            sections.extend(["", format_sources(results)])

        return "\n".join(sections)
    finally:
        if created_store is not None:
            close_store = getattr(created_store, "close", None)
            if callable(close_store):
                close_store()
        if created_parent_store is not None:
            created_parent_store.close()


def main(argv: Sequence[str] | None = None, retriever: Any | None = None) -> int:
    args = parse_args(argv)
    try:
        where = parse_where_filter(args.where)
        output = run_query(
            query=args.query,
            collection=args.collection,
            persist_path=args.persist_path,
            top_k=args.top_k,
            where=where,
            show_context=args.show_context,
            max_chars=args.max_chars,
            include_sources_summary=not args.no_sources_summary,
            chunk_mode=args.chunk_mode,
            parent_store_path=args.parent_store_path,
            show_child=args.show_child,
            show_parent_id=args.show_parent_id,
            retriever=retriever,
        )
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(output)
    return 0


def _validate_nonblank_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value.strip()


def _validate_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _clean_display_value(value: Any) -> str:
    if value is None:
        return UNKNOWN

    cleaned = str(value).strip()
    if not cleaned:
        return UNKNOWN
    return cleaned


def _format_distance(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, Real):
        return UNKNOWN
    return f"{float(value):.4f}"


def _result_text(result: Mapping[str, Any]) -> str:
    text = result.get("text")
    if not isinstance(text, str) or not text.strip():
        return "[empty text]"
    return text


def _display_source(result: Mapping[str, Any]) -> str:
    source_info = extract_source_info(result)
    for key in ("source", "relative_path", "filename"):
        value = source_info[key]
        if value != UNKNOWN:
            return value
    return UNKNOWN


def _format_context(retriever: Any, results: list[Mapping[str, Any]]) -> str:
    build_context = getattr(retriever, "build_context", None)
    if not callable(build_context):
        return "Context:\n[context unavailable: retriever has no build_context method]"

    try:
        context = build_context(results)
    except Exception as error:
        return f"Context:\n[context unavailable: {error}]"

    return f"Context:\n{context}"


def _parse_simple_where_object(
    where_text: str,
    original_error: json.JSONDecodeError,
) -> dict[str, Any]:
    text = where_text.strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ValueError(f"invalid JSON for --where: {original_error.msg}") from original_error

    body = text[1:-1].strip()
    if not body:
        return {}

    parsed = {}
    for item in body.split(","):
        if ":" not in item:
            raise ValueError(
                f"invalid JSON for --where: {original_error.msg}"
            ) from original_error

        key_text, value_text = item.split(":", 1)
        key = _strip_loose_json_token(key_text)
        value = _parse_loose_json_value(value_text)
        if not key:
            raise ValueError(
                f"invalid JSON for --where: {original_error.msg}"
            ) from original_error
        parsed[key] = value

    return parsed


def _parse_loose_json_value(value_text: str) -> Any:
    value = value_text.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return _strip_loose_json_token(value)


def _strip_loose_json_token(value: str) -> str:
    stripped = value.strip()
    if (
        len(stripped) >= 2
        and stripped[0] == stripped[-1]
        and stripped[0] in {"'", '"'}
    ):
        return stripped[1:-1]
    return stripped


def _resolved_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


if __name__ == "__main__":
    raise SystemExit(main())

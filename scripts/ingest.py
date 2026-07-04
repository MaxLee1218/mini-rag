from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunker import split_document
from app.config import VECTOR_COLLECTION_NAME, VECTOR_DB_PATH
from app.document_loader import load_document_file
from app.embeddings import Embedder
from app.vector_store import ChromaVectorStore


DEFAULT_EXTENSIONS = ".txt,.md"
IGNORED_DIR_NAMES = {".chroma", ".venv", ".git", "__pycache__", ".pytest_cache"}
SIMPLE_METADATA_TYPES = (str, int, float, bool)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load, chunk, embed, and store local documents."
    )
    parser.add_argument("--input", default="data/raw", help="Input file or directory.")
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
        "--reset",
        action="store_true",
        help="Reset the target collection before adding chunks.",
    )
    parser.add_argument(
        "--extensions",
        default=DEFAULT_EXTENSIONS,
        help="Comma-separated file extensions to ingest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and chunk documents without embedding or writing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional ingestion details.",
    )
    return parser.parse_args(argv)


def normalize_extensions(extensions: str | Iterable[str]) -> set[str]:
    if isinstance(extensions, str):
        extension_items = extensions.split(",")
    else:
        extension_items = extensions

    normalized = set()
    for extension in extension_items:
        clean_extension = str(extension).strip().lower()
        if not clean_extension:
            continue
        if not clean_extension.startswith("."):
            clean_extension = f".{clean_extension}"
        normalized.add(clean_extension)

    if not normalized:
        raise ValueError("at least one file extension is required")
    return normalized


def iter_input_files(
    input_path: str | Path,
    extensions: set[str],
) -> list[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"input path does not exist: {path}")

    normalized_extensions = normalize_extensions(extensions)
    if path.is_file():
        if path.suffix.lower() in normalized_extensions and not _is_ignored_file(
            path,
            path.parent,
        ):
            return [path]
        raise ValueError(f"no supported files found under input path: {path}")

    if not path.is_dir():
        raise ValueError(f"input path must be a file or directory: {path}")

    files = [
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in normalized_extensions
        and not _is_ignored_file(file_path, path)
    ]
    files.sort(key=lambda file_path: _relative_path(file_path, path).lower())

    if not files:
        raise ValueError(f"no supported files found under input path: {path}")
    return files


def ingest(
    input_path: str | Path = "data/raw",
    collection: str = VECTOR_COLLECTION_NAME,
    persist_path: str = VECTOR_DB_PATH,
    reset: bool = False,
    extensions: str | Iterable[str] = DEFAULT_EXTENSIONS,
    dry_run: bool = False,
    verbose: bool = False,
    embedder: Any | None = None,
    vector_store: Any | None = None,
    loader_func: Any | None = None,
    chunker_func: Any | None = None,
) -> dict[str, Any]:
    input_root = Path(input_path)
    normalized_extensions = normalize_extensions(extensions)
    input_files = iter_input_files(input_root, normalized_extensions)
    base_path = input_root if input_root.is_dir() else input_root.parent
    loader = loader_func if loader_func is not None else load_document_file
    chunker = chunker_func if chunker_func is not None else split_document

    prepared_chunks: list[dict[str, Any]] = []
    file_details = []

    for file_path in input_files:
        document = loader(file_path, base_path=base_path)
        document_text = _document_text(document)
        loader_metadata = _document_metadata(document)
        relative_path = _relative_path(file_path, base_path)
        chunker_document = {
            "text": document_text,
            "source": relative_path,
            "metadata": loader_metadata,
        }
        raw_chunks = list(chunker(chunker_document))
        if not raw_chunks:
            raise ValueError(f"chunker produced no chunks for: {file_path}")

        file_chunks = _prepare_chunks_for_embedding(
            raw_chunks=raw_chunks,
            file_path=file_path,
            relative_path=relative_path,
            loader_metadata=loader_metadata,
        )
        prepared_chunks.extend(file_chunks)
        file_details.append({"path": str(file_path), "chunks": len(file_chunks)})

    summary = {
        "input_path": str(input_root),
        "collection": collection,
        "persist_path": persist_path,
        "reset": bool(reset),
        "dry_run": bool(dry_run),
        "files_indexed": len(input_files),
        "chunks_created": len(prepared_chunks),
        "chunks_stored": 0,
        "file_details": file_details,
    }

    if dry_run:
        return summary

    active_embedder = embedder if embedder is not None else Embedder()
    active_store = (
        vector_store
        if vector_store is not None
        else ChromaVectorStore(collection_name=collection, persist_path=persist_path)
    )
    created_store = vector_store is None

    try:
        if reset:
            active_store.reset()
        embedded_chunks = active_embedder.embed_chunks(prepared_chunks)
        active_store.add_chunks(embedded_chunks)
        summary["chunks_stored"] = _stored_chunk_count(
            active_store,
            default_count=len(embedded_chunks),
        )
        return summary
    finally:
        if created_store:
            close_store = getattr(active_store, "close", None)
            if callable(close_store):
                close_store()


def print_summary(summary: Mapping[str, Any], verbose: bool = False) -> None:
    print(f"Input path: {summary['input_path']}")
    print(f"Collection: {summary['collection']}")
    print(f"Persist path: {summary['persist_path']}")
    print(f"Reset collection: {_yes_no(bool(summary['reset']))}")
    print(f"Dry run: {_yes_no(bool(summary['dry_run']))}")

    if verbose:
        for file_detail in summary.get("file_details", []):
            print(f"Indexed: {file_detail['path']} | chunks: {file_detail['chunks']}")

    print(f"Files indexed: {summary['files_indexed']}")
    print(f"Chunks created: {summary['chunks_created']}")
    print(f"Chunks stored: {summary['chunks_stored']}")

    if summary["dry_run"]:
        print("No embeddings generated.")
        print("No vector store writes performed.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = ingest(
            input_path=args.input,
            collection=args.collection,
            persist_path=args.persist_path,
            reset=args.reset,
            extensions=args.extensions,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_summary(summary, verbose=args.verbose)
    return 0


def _is_ignored_file(file_path: Path, input_root: Path) -> bool:
    parts = [part.lower() for part in file_path.parts]
    if set(parts).intersection(IGNORED_DIR_NAMES):
        return True

    for index, part in enumerate(parts[:-1]):
        if part == "data" and parts[index + 1] == "chroma":
            return True

    relative_path = _relative_path(file_path, input_root).lower()
    return relative_path == "data/chroma" or relative_path.startswith("data/chroma/")


def _relative_path(file_path: Path, base_path: Path) -> str:
    try:
        return file_path.relative_to(base_path).as_posix()
    except ValueError:
        return file_path.name


def _document_text(document: Any) -> str:
    if not isinstance(document, Mapping):
        raise ValueError("loaded document must be a dictionary")

    text = document.get("text", document.get("content"))
    if not isinstance(text, str) or not text.strip():
        raise ValueError("loaded document is empty")
    return text


def _document_metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError("document metadata must be a dictionary")

    merged = dict(metadata)
    if "source" in document:
        merged.setdefault("source", document["source"])
    return merged


def _prepare_chunks_for_embedding(
    raw_chunks: Sequence[Any],
    file_path: Path,
    relative_path: str,
    loader_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prepared_chunks = []

    for chunk_index, raw_chunk in enumerate(raw_chunks):
        chunk_text, chunk_metadata = _chunk_text_and_metadata(raw_chunk)
        ingest_metadata = {
            "source": relative_path,
            "filename": file_path.name,
            "relative_path": relative_path,
            "chunk_index": chunk_index,
        }
        metadata = {}
        metadata.update(loader_metadata)
        metadata.update(chunk_metadata)
        metadata.update(ingest_metadata)
        prepared_chunks.append(
            {
                "id": _chunk_id(
                    file_stem=file_path.stem,
                    relative_path=relative_path,
                    chunk_index=chunk_index,
                ),
                "text": chunk_text,
                "metadata": _sanitize_metadata(metadata),
            }
        )

    return prepared_chunks


def _chunk_text_and_metadata(raw_chunk: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(raw_chunk, str):
        chunk_text = raw_chunk
        metadata: dict[str, Any] = {}
    else:
        mapping = _chunk_mapping(raw_chunk)
        chunk_text = _chunk_text(mapping)
        metadata = _chunk_metadata(mapping)

    if not chunk_text.strip():
        raise ValueError("chunk text must not be blank")
    return chunk_text, metadata


def _chunk_mapping(raw_chunk: Any) -> Mapping[str, Any]:
    if is_dataclass(raw_chunk) and not isinstance(raw_chunk, type):
        return asdict(raw_chunk)
    if isinstance(raw_chunk, Mapping):
        return raw_chunk
    if hasattr(raw_chunk, "_asdict"):
        return raw_chunk._asdict()
    if hasattr(raw_chunk, "__dict__"):
        return vars(raw_chunk)
    raise ValueError("chunk must be a string, dictionary, or object with text")


def _chunk_text(chunk: Mapping[str, Any]) -> str:
    for text_key in ("text", "content", "chunk"):
        if text_key in chunk:
            text = chunk[text_key]
            if not isinstance(text, str):
                raise ValueError("chunk text must be a string")
            return text
    raise ValueError("chunk must contain text, content, or chunk")


def _chunk_metadata(chunk: Mapping[str, Any]) -> dict[str, Any]:
    metadata_value = chunk.get("metadata", {})
    if metadata_value is None:
        metadata: dict[str, Any] = {}
    elif isinstance(metadata_value, Mapping):
        metadata = dict(metadata_value)
    else:
        raise ValueError("chunk metadata must be a dictionary")

    ignored_keys = {"id", "text", "content", "chunk", "metadata"}
    for key, value in chunk.items():
        if key not in ignored_keys:
            metadata.setdefault(str(key), value)

    return metadata


def _chunk_id(file_stem: str, relative_path: str, chunk_index: int) -> str:
    safe_stem = _safe_file_stem(file_stem)
    path_hash = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:10]
    return f"{safe_stem}-{path_hash}-chunk-{chunk_index}"


def _safe_file_stem(file_stem: str) -> str:
    safe_stem = file_stem.strip().lower().replace(" ", "_")
    safe_stem = re.sub(r"[^a-z0-9_-]+", "_", safe_stem)
    safe_stem = re.sub(r"_+", "_", safe_stem).strip("_-")
    if not safe_stem:
        safe_stem = "document"
    return safe_stem[:48]


def _sanitize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            continue
        key = str(key)
        if isinstance(value, Path):
            sanitized[key] = str(value)
        elif isinstance(value, SIMPLE_METADATA_TYPES):
            sanitized[key] = value
        elif isinstance(value, (list, dict, tuple, set)):
            sanitized[key] = str(value)
        else:
            sanitized[key] = str(value)
    return sanitized


def _stored_chunk_count(vector_store: Any, default_count: int) -> int:
    count = getattr(vector_store, "count", None)
    if callable(count):
        return int(count())
    return default_count


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())

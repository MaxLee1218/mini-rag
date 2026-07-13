from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


class ParentStore(Protocol):
    """Persistent lookup contract for parent chunks."""

    def upsert(self, parents: Sequence[Mapping[str, Any]]) -> None: ...

    def get(self, parent_id: str) -> dict[str, Any] | None: ...

    def get_many(self, parent_ids: Sequence[str]) -> list[dict[str, Any]]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


class SQLiteParentStore:
    """SQLite-backed parent store with idempotent transactional upserts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._closed = False
        self._create_schema()

    def upsert(self, parents: Sequence[Mapping[str, Any]]) -> None:
        if isinstance(parents, (str, bytes)):
            raise ValueError("parents must be a sequence of parent records")
        records = [self._normalize_parent(parent) for parent in parents]
        if not records:
            return
        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO parents(parent_id, text, source, document_id, parent_index, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(parent_id) DO UPDATE SET
                    text=excluded.text,
                    source=excluded.source,
                    document_id=excluded.document_id,
                    parent_index=excluded.parent_index,
                    metadata_json=excluded.metadata_json
                """,
                records,
            )

    def get(self, parent_id: str) -> dict[str, Any] | None:
        clean_id = self._validate_parent_id(parent_id)
        row = self._connection.execute(
            "SELECT parent_id, text, metadata_json FROM parents WHERE parent_id = ?",
            (clean_id,),
        ).fetchone()
        return None if row is None else self._row_to_parent(row)

    def get_many(self, parent_ids: Sequence[str]) -> list[dict[str, Any]]:
        if isinstance(parent_ids, (str, bytes)):
            raise ValueError("parent_ids must be a sequence of strings")
        ids = [self._validate_parent_id(parent_id) for parent_id in parent_ids]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._connection.execute(
            f"SELECT parent_id, text, metadata_json FROM parents WHERE parent_id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {row[0]: self._row_to_parent(row) for row in rows}
        return [by_id[parent_id] for parent_id in ids if parent_id in by_id]

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM parents").fetchone()
        return int(row[0])

    def reset(self) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM parents")

    def close(self) -> None:
        if self._closed:
            return
        self._connection.close()
        self._closed = True

    def __enter__(self) -> SQLiteParentStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS parents(
                    parent_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    parent_index INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

    def _normalize_parent(self, parent: Mapping[str, Any]) -> tuple[Any, ...]:
        if not isinstance(parent, Mapping):
            raise ValueError("each parent must be a dictionary")
        parent_id = self._validate_parent_id(parent.get("id"))
        text = parent.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"parent text must not be blank: {parent_id}")
        metadata = parent.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"parent metadata must be a dictionary: {parent_id}")
        source = metadata.get("source")
        document_id = metadata.get("document_id")
        parent_index = metadata.get("parent_index")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"parent metadata.source must not be blank: {parent_id}")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"parent metadata.document_id must not be blank: {parent_id}")
        if isinstance(parent_index, bool) or not isinstance(parent_index, int):
            raise ValueError(f"parent metadata.parent_index must be an integer: {parent_id}")
        return (
            parent_id,
            text,
            source,
            document_id,
            parent_index,
            json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True),
        )

    def _validate_parent_id(self, parent_id: Any) -> str:
        if not isinstance(parent_id, str) or not parent_id.strip():
            raise ValueError("parent_id must not be blank")
        return parent_id.strip()

    def _row_to_parent(self, row: tuple[Any, ...]) -> dict[str, Any]:
        return {"id": row[0], "text": row[1], "metadata": json.loads(row[2])}

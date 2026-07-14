from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from app.faq.models import FAQImportSummary, FAQRecord
from app.faq.text import normalize_question


logger = logging.getLogger(__name__)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS faqs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    source TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS faq_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    faq_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    FOREIGN KEY (faq_id) REFERENCES faqs(id) ON DELETE CASCADE,
    UNIQUE(faq_id, normalized_alias)
);

CREATE TABLE IF NOT EXISTS faq_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class FAQRepository:
    """Persist maintained FAQ records in a small SQLite database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def ensure_schema(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.executescript(_SCHEMA)
                connection.execute(
                    "INSERT OR IGNORE INTO faq_meta(key, value) VALUES (?, ?)",
                    ("index_version", "0"),
                )
        except Exception:
            logger.exception(
                "faq_repository_schema_initialization_failed",
                extra={"error_category": "sqlite_initialization"},
            )
            raise

    def list_enabled(self) -> list[FAQRecord]:
        return self._list_records(enabled_only=True)

    def list_all(self) -> list[FAQRecord]:
        return self._list_records(enabled_only=False)

    def get_index_version(self) -> int:
        self.ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM faq_meta WHERE key = ?", ("index_version",)
            ).fetchone()
        return int(row["value"]) if row is not None else 0

    def import_records(
        self, records: Sequence[FAQRecord]
    ) -> FAQImportSummary:
        self.ensure_schema()
        prepared = [_validated_record(record) for record in records]
        inserted = updated = unchanged = 0
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            for record in prepared:
                current = self._get_record(connection, record.id)
                if current == record:
                    unchanged += 1
                    continue

                if current is None:
                    inserted += 1
                    connection.execute(
                        """
                        INSERT INTO faqs(
                            id, question, answer, source, enabled,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.id,
                            record.question,
                            record.answer,
                            record.source,
                            int(record.enabled),
                            now,
                            now,
                        ),
                    )
                else:
                    updated += 1
                    connection.execute(
                        """
                        UPDATE faqs
                        SET question = ?, answer = ?, source = ?, enabled = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            record.question,
                            record.answer,
                            record.source,
                            int(record.enabled),
                            now,
                            record.id,
                        ),
                    )
                    connection.execute(
                        "DELETE FROM faq_aliases WHERE faq_id = ?", (record.id,)
                    )

                connection.executemany(
                    """
                    INSERT INTO faq_aliases(faq_id, alias, normalized_alias)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (record.id, alias, normalize_question(alias))
                        for alias in record.aliases
                    ],
                )

            row = connection.execute(
                "SELECT value FROM faq_meta WHERE key = ?", ("index_version",)
            ).fetchone()
            index_version = int(row["value"])
            if inserted + updated:
                index_version += 1
                connection.execute(
                    "UPDATE faq_meta SET value = ? WHERE key = ?",
                    (str(index_version), "index_version"),
                )

        return FAQImportSummary(
            inserted=inserted,
            updated=updated,
            unchanged=unchanged,
            index_version=index_version,
        )

    def _list_records(self, *, enabled_only: bool) -> list[FAQRecord]:
        self.ensure_schema()
        query = "SELECT * FROM faqs"
        parameters: tuple[int, ...] = ()
        if enabled_only:
            query += " WHERE enabled = ?"
            parameters = (1,)
        query += " ORDER BY id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
            return [self._record_from_row(connection, row) for row in rows]

    def _get_record(
        self, connection: sqlite3.Connection, faq_id: str
    ) -> FAQRecord | None:
        row = connection.execute(
            "SELECT * FROM faqs WHERE id = ?", (faq_id,)
        ).fetchone()
        if row is None:
            return None
        return self._record_from_row(connection, row)

    def _record_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> FAQRecord:
        aliases = connection.execute(
            "SELECT alias FROM faq_aliases WHERE faq_id = ? ORDER BY id", (row["id"],)
        ).fetchall()
        return FAQRecord(
            id=row["id"],
            question=row["question"],
            answer=row["answer"],
            aliases=tuple(alias["alias"] for alias in aliases),
            source=row["source"],
            enabled=bool(row["enabled"]),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _validated_record(record: FAQRecord) -> FAQRecord:
    if not isinstance(record, FAQRecord):
        raise ValueError("records must contain FAQRecord values")
    if not isinstance(record.enabled, bool):
        raise ValueError("FAQ enabled must be a boolean")
    faq_id = _required_text(record.id, "FAQ id")
    question = _required_text(record.question, "FAQ question")
    answer = _required_text(record.answer, "FAQ answer")
    if not isinstance(record.aliases, tuple):
        raise ValueError("FAQ aliases must be a tuple")
    aliases: list[str] = []
    seen: set[str] = set()
    for alias in record.aliases:
        clean_alias = _required_text(alias, "FAQ alias")
        normalized = normalize_question(clean_alias)
        if normalized in seen:
            continue
        seen.add(normalized)
        aliases.append(clean_alias)
    if record.source is not None and not isinstance(record.source, str):
        raise ValueError("FAQ source must be a string or None")
    source = record.source.strip() if isinstance(record.source, str) else None
    return FAQRecord(
        id=faq_id,
        question=question,
        answer=answer,
        aliases=tuple(aliases),
        source=source or None,
        enabled=record.enabled,
    )


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value.strip()

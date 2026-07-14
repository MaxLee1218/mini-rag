from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.config import FAQ_DB_PATH
from app.faq.models import FAQRecord
from app.faq.repository import FAQRepository


_ALLOWED_FIELDS = {
    "id",
    "question",
    "answer",
    "aliases",
    "source",
    "enabled",
}
_REQUIRED_FIELDS = {"id", "question", "answer"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the local FAQ database.")
    parser.add_argument("--db-path", type=Path, default=FAQ_DB_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="Create the FAQ schema.")
    import_parser = subparsers.add_parser("import", help="Import FAQ JSON.")
    import_parser.add_argument("path", type=Path)
    subparsers.add_parser("list", help="List FAQ metadata.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repository = FAQRepository(args.db_path)
    try:
        if args.command == "init":
            repository.ensure_schema()
            print(f"initialized {args.db_path}")
        elif args.command == "import":
            records = _load_records(args.path)
            summary = repository.import_records(records)
            print(
                f"inserted={summary.inserted} updated={summary.updated} "
                f"unchanged={summary.unchanged} "
                f"index_version={summary.index_version}"
            )
        elif args.command == "list":
            for record in repository.list_all():
                print(
                    f"{record.id}\t{record.question}\t"
                    f"enabled={str(record.enabled).lower()}\t"
                    f"aliases={len(record.aliases)}"
                )
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


def _load_records(path: Path) -> list[FAQRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("FAQ import root must be a JSON array")
    return [_record_from_json(item) for item in payload]


def _record_from_json(item: Any) -> FAQRecord:
    if not isinstance(item, dict):
        raise ValueError("each FAQ must be a JSON object")
    unknown = set(item) - _ALLOWED_FIELDS
    missing = _REQUIRED_FIELDS - set(item)
    if unknown:
        raise ValueError(f"unknown FAQ fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing FAQ fields: {', '.join(sorted(missing))}")
    aliases = item.get("aliases", [])
    if not isinstance(aliases, list) or not all(
        isinstance(alias, str) for alias in aliases
    ):
        raise ValueError("FAQ aliases must be an array of strings")
    enabled = item.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("FAQ enabled must be a boolean")
    source = item.get("source")
    if source is not None and not isinstance(source, str):
        raise ValueError("FAQ source must be a string or null")
    for field in _REQUIRED_FIELDS:
        if not isinstance(item[field], str) or not item[field].strip():
            raise ValueError(f"FAQ {field} must not be blank")
    return FAQRecord(
        id=item["id"],
        question=item["question"],
        answer=item["answer"],
        aliases=tuple(aliases),
        source=source,
        enabled=enabled,
    )


if __name__ == "__main__":
    raise SystemExit(main())

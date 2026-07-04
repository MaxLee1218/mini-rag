from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_SYSTEM_PROMPT = """你是一个严谨的 RAG 问答助手。
你必须只基于上下文区块回答问题区块，不要使用外部知识。
一般情况下，请使用与用户问题相同的语言回答。
优先级规则：
1. 如果上下文区块不足以回答问题，必须只输出：Not found in knowledge base.
不要输出“答案：”，不要输出“来源：”，不要输出任何解释。
不要翻译成中文。
不要输出“根据当前资料无法确定”等其他表达。
2. 如果使用了上下文区块中的信息回答问题，每个关键结论后必须引用上下文编号，例如 [1]、[2]。
回答末尾必须包含“来源：”部分。
“来源：”部分只能列出实际使用过的上下文编号和对应 Source。
3. 只能引用上下文区块中存在的编号。
4. 不要编造不存在的事实、数字、编号、文件名、路径或来源。"""

DEFAULT_NO_CONTEXT_MESSAGE = "No relevant context was retrieved."
TRUNCATION_MESSAGE = "[Context truncated due to length limit.]"
NOT_FOUND_MESSAGE = "Not found in knowledge base."
SOURCES_SECTION_PATTERN = re.compile(r"^\s*(来源：|Sources:)\s*$")
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

TEXT_FIELDS = ("text", "content", "page_content", "chunk")
SOURCE_FIELDS = ("source", "source_file", "file_path", "path", "filename", "title")
CHUNK_FIELDS = ("chunk_id", "chunk_index", "chunk_no")
METADATA_FIELDS = (
    *SOURCE_FIELDS,
    "page",
    *CHUNK_FIELDS,
    "id",
    "score",
    "distance",
)


def build_prompt(
    question: str,
    contexts: Sequence[Any],
    *,
    max_context_chars: int = 12000,
    max_chunks: int | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Build a deterministic RAG prompt from a question and retrieved contexts."""
    clean_question = _validate_nonblank_string(question, "question")
    context_char_limit = _validate_positive_int(
        max_context_chars,
        "max_context_chars",
    )
    chunk_limit = _validate_optional_positive_int(max_chunks, "max_chunks")
    clean_system_prompt = _validate_nonblank_string(system_prompt, "system_prompt")

    formatted_context = _format_contexts(
        contexts,
        max_context_chars=context_char_limit,
        max_chunks=chunk_limit,
    )
    answer_instructions = _answer_instructions(
        has_context=formatted_context != DEFAULT_NO_CONTEXT_MESSAGE
    )

    return "\n\n".join(
        [
            f"<system>\n{clean_system_prompt}\n</system>",
            f"<context>\n{formatted_context}\n</context>",
            f"<question>\n{clean_question}\n</question>",
            f"<answer_instructions>\n{answer_instructions}\n</answer_instructions>",
        ]
    )


def build_sources_section(
    contexts: Sequence[Any],
    *,
    cited_indices: set[int] | None = None,
    max_sources: int | None = None,
) -> str:
    """Build a deterministic sources section from retrieved contexts."""
    source_limit = _validate_optional_positive_int(max_sources, "max_sources")
    valid_cited_indices = _valid_cited_indices(cited_indices)
    lines = []

    for entry in _iter_valid_context_entries(contexts):
        index = entry["index"]
        if valid_cited_indices is not None and index not in valid_cited_indices:
            continue

        lines.append(f"- [{index}] {entry['source']}")
        if source_limit is not None and len(lines) >= source_limit:
            break

    if not lines:
        return ""
    return "来源：\n" + "\n".join(lines)


def extract_cited_indices(answer: str) -> set[int]:
    """Extract positive citation indices like [1] and [10] from an answer."""
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")

    indices = set()
    for match in CITATION_PATTERN.finditer(answer):
        index = int(match.group(1))
        if index > 0:
            indices.add(index)
    return indices


def append_sources_to_answer(
    answer: str,
    contexts: Sequence[Any],
    *,
    max_sources: int | None = None,
    only_cited: bool = True,
) -> str:
    """Append a context-derived sources section without trusting model output."""
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    source_limit = _validate_optional_positive_int(max_sources, "max_sources")

    if not answer.strip():
        return answer
    if answer.strip() == NOT_FOUND_MESSAGE:
        return NOT_FOUND_MESSAGE

    answer_without_sources = _strip_existing_sources_section(answer)
    cited_indices = extract_cited_indices(answer_without_sources) if only_cited else None

    sources_section = build_sources_section(
        contexts,
        cited_indices=cited_indices if cited_indices else None,
        max_sources=source_limit,
    )
    if not sources_section:
        return answer_without_sources.rstrip()

    return f"{answer_without_sources.rstrip()}\n\n{sources_section}"


class PromptBuilder:
    """Reusable prompt builder with fixed prompt options."""

    def __init__(
        self,
        *,
        max_context_chars: int = 12000,
        max_chunks: int | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.max_context_chars = _validate_positive_int(
            max_context_chars,
            "max_context_chars",
        )
        self.max_chunks = _validate_optional_positive_int(max_chunks, "max_chunks")
        self.system_prompt = _validate_nonblank_string(
            system_prompt,
            "system_prompt",
        )

    def build(self, question: str, contexts: Sequence[Any]) -> str:
        """Build a prompt using this builder's configured limits."""
        return build_prompt(
            question,
            contexts,
            max_context_chars=self.max_context_chars,
            max_chunks=self.max_chunks,
            system_prompt=self.system_prompt,
        )


def _format_contexts(
    contexts: Sequence[Any],
    *,
    max_context_chars: int,
    max_chunks: int | None,
) -> str:
    blocks = [
        _format_context_block(
            index=entry["index"],
            text=entry["text"],
            metadata=entry["metadata"],
        )
        for entry in _iter_valid_context_entries(contexts, max_chunks=max_chunks)
    ]

    if not blocks:
        return DEFAULT_NO_CONTEXT_MESSAGE

    formatted = "\n\n".join(blocks)
    return _truncate_context(formatted, max_context_chars)


def _iter_valid_context_entries(
    contexts: Sequence[Any],
    *,
    max_chunks: int | None = None,
) -> list[dict[str, Any]]:
    if isinstance(contexts, (str, bytes)):
        raise ValueError("contexts must be a sequence of context items")

    entries = []
    for item in contexts:
        text = _extract_text(item)
        if not text:
            continue

        metadata = _extract_metadata(item)
        entries.append(
            {
                "index": len(entries) + 1,
                "item": item,
                "text": text,
                "metadata": metadata,
                "source": _source_from_metadata(metadata),
            }
        )
        if max_chunks is not None and len(entries) >= max_chunks:
            break

    return entries


def _extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, Mapping):
        for field_name in TEXT_FIELDS:
            text = _clean_text_value(item.get(field_name))
            if text:
                return text
        return ""

    for field_name in TEXT_FIELDS:
        text = _clean_text_value(getattr(item, field_name, None))
        if text:
            return text
    return ""


def _extract_metadata(item: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    if isinstance(item, str):
        return metadata

    if isinstance(item, Mapping):
        nested_metadata = item.get("metadata")
        if isinstance(nested_metadata, Mapping):
            metadata.update(nested_metadata)
        _merge_known_fields(metadata, item)
        return metadata

    nested_metadata = getattr(item, "metadata", None)
    if isinstance(nested_metadata, Mapping):
        metadata.update(nested_metadata)

    for field_name in METADATA_FIELDS:
        if hasattr(item, field_name):
            value = getattr(item, field_name)
            if _display_value(value) is not None:
                metadata[field_name] = value

    return metadata


def _format_context_block(
    *,
    index: int,
    text: str,
    metadata: Mapping[str, Any],
) -> str:
    lines = [f"[{index}]", f"Source: {_source_from_metadata(metadata)}"]

    chunk_id = _first_display_value(metadata, CHUNK_FIELDS)
    if chunk_id is not None:
        lines.append(f"Chunk ID: {chunk_id}")

    item_id = _display_value(metadata.get("id"))
    if item_id is not None:
        lines.append(f"ID: {item_id}")

    page = _display_value(metadata.get("page"))
    if page is not None:
        lines.append(f"Page: {page}")

    score = _display_value(metadata.get("score"))
    if score is not None:
        lines.append(f"Score: {score}")

    distance = _display_value(metadata.get("distance"))
    if distance is not None:
        lines.append(f"Distance: {distance}")

    lines.extend(["", text])
    return "\n".join(lines)


def _truncate_context(context: str, max_context_chars: int) -> str:
    if len(context) <= max_context_chars:
        return context

    if max_context_chars <= len(TRUNCATION_MESSAGE):
        return TRUNCATION_MESSAGE

    prefix_limit = max_context_chars - len(TRUNCATION_MESSAGE) - 1
    if prefix_limit <= 0:
        return TRUNCATION_MESSAGE

    prefix = _trim_to_readable_boundary(context[:prefix_limit]).rstrip()
    if not prefix:
        return TRUNCATION_MESSAGE
    return f"{prefix}\n{TRUNCATION_MESSAGE}"


def _trim_to_readable_boundary(text: str) -> str:
    chunk_boundary = text.rfind("\n\n[")
    if chunk_boundary > 0:
        return text[:chunk_boundary]

    for boundary in ("\n", "。", ". "):
        boundary_index = text.rfind(boundary)
        if boundary_index > 0:
            end = boundary_index + len(boundary.rstrip())
            return text[:end]

    return text


def _answer_instructions(*, has_context: bool) -> str:
    lines = [
        "你必须只基于 <context> 回答 <question>。",
        "一般情况下，请使用与用户问题相同的语言回答。",
        "优先级 1：如果 <context> 不足以回答问题，必须只输出：Not found in knowledge base.",
        "不要输出“答案：”，不要输出“来源：”，不要输出任何解释。",
        "不要翻译成中文。",
        "不要输出“根据当前资料无法确定”等其他表达。",
        "优先级 2：如果使用了 <context> 中的信息回答问题，每个关键结论后必须引用编号，例如 [1]、[2]。",
        "回答末尾必须输出“来源：”部分。",
        "“来源：”部分只能列出实际使用过的上下文编号和对应 Source。",
        "不允许引用 <context> 中不存在的编号。",
        "不要编造不存在的事实、数字、编号、文件名、路径或来源。",
    ]
    if not has_context:
        lines.append("当前没有检索到相关资料，不能编造答案。")
    return "\n".join(lines)


def _strip_existing_sources_section(answer: str) -> str:
    lines = answer.splitlines()
    for index, line in enumerate(lines):
        if SOURCES_SECTION_PATTERN.match(line):
            return "\n".join(lines[:index])
    return answer


def _valid_cited_indices(cited_indices: set[int] | None) -> set[int] | None:
    if cited_indices is None:
        return None

    valid_indices = set()
    for index in cited_indices:
        if isinstance(index, bool):
            continue
        if isinstance(index, int) and index > 0:
            valid_indices.add(index)
    return valid_indices


def _source_from_metadata(metadata: Mapping[str, Any]) -> str:
    return _first_display_value(metadata, SOURCE_FIELDS) or "unknown"


def _first_display_value(
    values: Mapping[str, Any],
    field_names: Sequence[str],
) -> str | None:
    for field_name in field_names:
        value = _display_value(values.get(field_name))
        if value is not None:
            return value
    return None


def _merge_known_fields(
    metadata: dict[str, Any],
    item: Mapping[str, Any],
) -> None:
    for field_name in METADATA_FIELDS:
        if field_name in item and _display_value(item[field_name]) is not None:
            metadata[field_name] = item[field_name]


def _clean_text_value(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _display_value(value: Any) -> str | None:
    if value is None:
        return None
    if callable(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return value
    return str(value)


def _validate_nonblank_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value.strip()


def _validate_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _validate_optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _validate_positive_int(value, field_name)

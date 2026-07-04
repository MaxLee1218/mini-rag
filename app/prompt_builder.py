from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_SYSTEM_PROMPT = """你是一个严谨的 RAG 问答助手。
你必须只根据给定的上下文回答问题，不要使用外部知识。
不要编造不存在的事实、数字、文件名或来源。
回答时尽量引用上下文编号，例如 [1]、[2]。
一般情况下，请使用与用户问题相同的语言回答。
如果上下文不足以回答问题，必须只输出：Not found in knowledge base.
不要附加解释。
不要翻译成中文。
不要输出“根据当前资料无法确定”等其他表达。"""

DEFAULT_NO_CONTEXT_MESSAGE = "No relevant context was retrieved."
TRUNCATION_MESSAGE = "[Context truncated due to length limit.]"

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
    if isinstance(contexts, (str, bytes)):
        raise ValueError("contexts must be a sequence of context items")

    blocks = []
    for item in contexts:
        text = _extract_text(item)
        if not text:
            continue

        blocks.append(
            _format_context_block(
                index=len(blocks) + 1,
                text=text,
                metadata=_extract_metadata(item),
            )
        )
        if max_chunks is not None and len(blocks) >= max_chunks:
            break

    if not blocks:
        return DEFAULT_NO_CONTEXT_MESSAGE

    formatted = "\n\n".join(blocks)
    return _truncate_context(formatted, max_context_chars)


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
        "请基于 <context> 回答 <question>。",
        "一般情况下，请使用与用户问题相同的语言回答。",
        "如果上下文不足以回答问题，必须只输出：Not found in knowledge base.",
        "不要附加解释。",
        "不要翻译成中文。",
        "不要输出“根据当前资料无法确定”等其他表达。",
        "不要编造不存在的事实、数字、文件名或来源。",
        "回答中如使用某个上下文片段的信息，请引用对应编号，例如 [1]。",
    ]
    if not has_context:
        lines.append("当前没有检索到相关资料，不能编造答案。")
    return "\n".join(lines)


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

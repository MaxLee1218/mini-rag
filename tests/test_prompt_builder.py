from dataclasses import dataclass

import pytest

from app.prompt_builder import (
    DEFAULT_NO_CONTEXT_MESSAGE,
    DEFAULT_SYSTEM_PROMPT,
    PromptBuilder,
    build_prompt,
)


FALLBACK_ANSWER = "Not found in knowledge base."
TRUNCATION_MESSAGE = "[Context truncated due to length limit.]"


def extract_context(prompt: str) -> str:
    start = prompt.index("<context>") + len("<context>")
    end = prompt.index("</context>")
    return prompt[start:end].strip()


def test_build_prompt_formats_dict_context_with_source_and_instructions():
    prompt = build_prompt(
        "什么是 RAG？",
        [
            {
                "text": "RAG 是 Retrieval-Augmented Generation 的缩写。",
                "metadata": {"source": "docs/rag.md", "chunk_id": 1},
                "score": 0.91,
            }
        ],
    )

    assert "什么是 RAG？" in prompt
    assert "RAG 是 Retrieval-Augmented Generation 的缩写。" in prompt
    assert "Source: docs/rag.md" in prompt
    assert "Chunk ID: 1" in prompt
    assert "Score: 0.91" in prompt
    assert "[1]" in prompt
    assert DEFAULT_SYSTEM_PROMPT in prompt
    assert "不要编造" in prompt
    assert FALLBACK_ANSWER in prompt
    assert "不要翻译成中文" in prompt


def test_build_prompt_handles_empty_contexts_without_inventing_answer():
    prompt = build_prompt("未知问题", [])

    assert "未知问题" in prompt
    assert DEFAULT_NO_CONTEXT_MESSAGE in prompt
    assert "当前没有检索到相关资料" in prompt
    assert FALLBACK_ANSWER in prompt


def test_build_prompt_accepts_string_context_with_unknown_source():
    prompt = build_prompt("问题", ["这是一个纯字符串上下文。"])

    assert "这是一个纯字符串上下文。" in prompt
    assert "[1]" in prompt
    assert "Source: unknown" in prompt


def test_extract_text_uses_first_non_empty_text_field():
    prompt = build_prompt(
        "问题",
        [{"text": "  ", "content": "有效内容", "page_content": "备用内容"}],
    )

    assert "有效内容" in prompt
    assert "备用内容" not in prompt


def test_build_prompt_accepts_alternate_text_and_source_fields():
    prompt = build_prompt(
        "问题",
        [
            {"content": "来自 content 的文本", "source_file": "a.md"},
            {"page_content": "来自 page_content 的文本", "file_path": "b.md"},
        ],
    )

    assert "来自 content 的文本" in prompt
    assert "Source: a.md" in prompt
    assert "来自 page_content 的文本" in prompt
    assert "Source: b.md" in prompt


def test_top_level_metadata_overrides_nested_metadata_values():
    prompt = build_prompt(
        "问题",
        [
            {
                "text": "带冲突元数据的文本",
                "metadata": {
                    "source": "old.md",
                    "score": 0.1,
                    "distance": 9.9,
                },
                "source": "new.md",
                "score": 0.8,
                "distance": 0.2,
            }
        ],
    )

    assert "Source: new.md" in prompt
    assert "Score: 0.8" in prompt
    assert "Distance: 0.2" in prompt
    assert "old.md" not in prompt
    assert "Score: 0.1" not in prompt
    assert "Distance: 9.9" not in prompt


def test_retriever_result_id_is_displayed_as_id_not_chunk_id():
    prompt = build_prompt(
        "问题",
        [
            {
                "id": "vector-id-1",
                "text": "retriever result text",
                "metadata": {"source": "docs/retriever.md"},
                "distance": 0.1234,
            }
        ],
    )
    context = extract_context(prompt)

    assert "ID: vector-id-1" in context
    assert "Distance: 0.1234" in context
    assert "Chunk ID: vector-id-1" not in context


@pytest.mark.parametrize(
    "chunk_field",
    ["chunk_id", "chunk_index", "chunk_no"],
)
def test_explicit_chunk_fields_are_displayed_as_chunk_id(chunk_field):
    prompt = build_prompt(
        "问题",
        [{"text": "chunk text", "source": "chunk.md", chunk_field: 7}],
    )

    assert "Chunk ID: 7" in prompt


def test_build_prompt_limits_valid_contexts_with_max_chunks():
    prompt = build_prompt(
        "问题",
        [
            {"text": "first"},
            {"text": "second"},
            {"text": "third"},
        ],
        max_chunks=2,
    )

    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "[3]" not in prompt
    assert "first" in prompt
    assert "second" in prompt
    assert "third" not in prompt


def test_build_prompt_truncates_context_within_max_context_chars():
    prompt = build_prompt(
        "问题",
        [{"text": "alpha " * 100, "source": "long.md"}],
        max_context_chars=120,
    )
    context = extract_context(prompt)

    assert len(context) <= 120
    assert TRUNCATION_MESSAGE in context


def test_build_prompt_keeps_truncation_message_when_limit_is_tiny():
    prompt = build_prompt(
        "问题",
        [{"text": "alpha " * 100, "source": "long.md"}],
        max_context_chars=5,
    )
    context = extract_context(prompt)

    assert context == TRUNCATION_MESSAGE


@pytest.mark.parametrize("question", ["", "   "])
def test_build_prompt_rejects_blank_question(question):
    with pytest.raises(ValueError, match="question must not be blank"):
        build_prompt(question, [])


@pytest.mark.parametrize("max_context_chars", [0, -1, True])
def test_build_prompt_rejects_invalid_max_context_chars(max_context_chars):
    with pytest.raises(ValueError, match="max_context_chars must be a positive integer"):
        build_prompt("问题", [], max_context_chars=max_context_chars)


@pytest.mark.parametrize("max_chunks", [0, -1, True])
def test_build_prompt_rejects_invalid_max_chunks(max_chunks):
    with pytest.raises(ValueError, match="max_chunks must be a positive integer"):
        build_prompt("问题", [], max_chunks=max_chunks)


@dataclass
class ObjectContext:
    content: str
    metadata: dict
    source: str
    page: int


def test_build_prompt_accepts_object_context_and_top_level_attributes_win():
    prompt = build_prompt(
        "问题",
        [
            ObjectContext(
                content="对象上下文",
                metadata={"source": "old-object.md", "page": 1},
                source="new-object.md",
                page=3,
            )
        ],
    )

    assert "对象上下文" in prompt
    assert "Source: new-object.md" in prompt
    assert "Page: 3" in prompt
    assert "old-object.md" not in prompt


def test_prompt_builder_class_uses_configured_values():
    builder = PromptBuilder(max_context_chars=90, max_chunks=1, system_prompt="系统")

    prompt = builder.build(
        "问题",
        [
            {"text": "alpha", "source": "a.md"},
            {"text": "beta", "source": "b.md"},
        ],
    )

    assert "<system>\n系统\n</system>" in prompt
    assert "alpha" in prompt
    assert "beta" not in prompt

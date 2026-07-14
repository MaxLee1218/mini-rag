from dataclasses import replace

import pytest

from app.faq.matcher import FAQMatcher
from app.faq.models import FAQRecord
from app.faq.repository import FAQRepository
from app.faq.text import normalize_question, tokenize_question


def _password_record(**changes):
    record = FAQRecord(
        id="faq-password",
        question="如何重置密码？",
        aliases=("忘记密码怎么办", "密码找回方法"),
        answer="请从账户安全设置重置密码。",
        source="account.md",
    )
    return replace(record, **changes)


def test_normalization_is_stable_for_nfkc_case_spacing_and_punctuation():
    assert normalize_question("  什么是 ＲＡＧ？？ ") == "什么是 rag"
    assert normalize_question(" Redis   IS Fast! ") == "redis is fast"
    assert normalize_question("") == ""


def test_tokenizer_supports_chinese_english_and_numbers():
    tokens = tokenize_question("RAG 使用 Redis 7 缓存")

    assert "rag" in tokens
    assert "redis" in tokens
    assert "7" in tokens
    assert any("缓存" in token or token in {"缓", "存"} for token in tokens)


def test_canonical_and_alias_exact_matches_are_distinguished():
    matcher = FAQMatcher([_password_record()], threshold=1.0, margin=0.15)

    canonical = matcher.match("如何重置密码？？")
    alias = matcher.match("  忘记密码怎么办！ ")

    assert canonical is not None and canonical.match_type == "exact"
    assert alias is not None and alias.match_type == "alias"
    assert canonical.faq_id == alias.faq_id == "faq-password"
    assert canonical.answer == "请从账户安全设置重置密码。"


def test_non_exact_chinese_surface_matches_same_faq():
    matcher = FAQMatcher([_password_record()], threshold=0.0, margin=0.0)

    match = matcher.match("我忘记密码了，应该怎么找回？")

    assert match is not None
    assert match.faq_id == "faq-password"
    assert match.match_type == "bm25"


def test_same_query_hits_low_threshold_and_misses_extreme_threshold():
    query = "忘记密码如何找回"

    low = FAQMatcher([_password_record()], threshold=0.0, margin=0.0)
    high = FAQMatcher(
        [_password_record()], threshold=1_000_000.0, margin=0.0
    )

    assert low.match(query) is not None
    assert high.match(query) is None


def test_margin_uses_second_distinct_faq_not_second_alias():
    records = [
        _password_record(aliases=("密码问题", "密码遇到问题")),
        FAQRecord(
            id="faq-pin",
            question="如何修改支付密码？",
            aliases=("支付密码问题",),
            answer="请在支付安全设置修改。",
        ),
    ]

    strict = FAQMatcher(records, threshold=0.0, margin=10_000.0)
    relaxed = FAQMatcher(records, threshold=0.0, margin=0.0)

    assert strict.match("密码出现问题") is None
    assert relaxed.match("密码出现问题") is not None


def test_empty_index_blank_query_and_unrelated_query_are_misses():
    assert FAQMatcher([], threshold=0.0, margin=0.0).match("问题") is None
    matcher = FAQMatcher([_password_record()], threshold=0.0, margin=0.0)
    assert matcher.match("  ") is None
    assert matcher.match("火星天气") is None


@pytest.mark.parametrize(
    ("threshold", "margin"), [(-1.0, 0.0), (0.0, -0.1)]
)
def test_matcher_rejects_negative_threshold_or_margin(threshold, margin):
    with pytest.raises(ValueError):
        FAQMatcher([_password_record()], threshold=threshold, margin=margin)


def test_disabled_repository_record_does_not_enter_matcher(tmp_path):
    repository = FAQRepository(tmp_path / "faq.db")
    repository.import_records([_password_record(enabled=False)])
    matcher = FAQMatcher(
        repository.list_enabled(), threshold=0.0, margin=0.0
    )

    assert matcher.match("如何重置密码？") is None


def test_ambiguous_exact_surface_fails_closed():
    records = [
        _password_record(aliases=("账户问题",)),
        FAQRecord("faq-account", "账户问题", "请查看账户帮助。"),
    ]
    matcher = FAQMatcher(records, threshold=0.0, margin=0.0)

    assert matcher.match("账户问题") is None

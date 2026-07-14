import json

from app.faq.cache import NullFAQCache, RedisFAQCache, faq_cache_key
from app.faq.models import FAQMatch


def _match():
    return FAQMatch(
        faq_id="faq-rag",
        question="什么是 RAG？",
        answer="RAG 是检索增强生成。",
        source="README.md",
        score=3.5,
        match_type="bm25",
    )


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}
        self.deleted = []

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value
        self.ttls[key] = ttl

    def delete(self, key):
        self.deleted.append(key)
        self.values.pop(key, None)


class FailingRedis:
    def get(self, key):
        raise TimeoutError("redis unavailable at redis://secret@localhost")

    def setex(self, key, ttl, value):
        raise ConnectionError("redis unavailable")

    def delete(self, key):
        raise ConnectionError("redis unavailable")


def test_cache_key_is_versioned_hashed_and_value_is_json():
    client = FakeRedis()
    cache = RedisFAQCache(client, ttl_seconds=60)

    cache.set("什么是 rag", 7, _match())

    key = next(iter(client.values))
    assert key.startswith("mini-rag:faq:v7:query:")
    assert "什么是 rag" not in key
    assert client.ttls[key] == 60
    assert json.loads(client.values[key])["faq_id"] == "faq-rag"
    cached = cache.get("什么是 rag", 7)
    assert cached is not None
    assert cached.faq_id == "faq-rag"
    assert cached.score == 3.5
    assert cached.match_type == "cache"


def test_cache_version_change_does_not_read_old_value():
    client = FakeRedis()
    cache = RedisFAQCache(client, ttl_seconds=60)
    cache.set("query", 1, _match())

    assert cache.get("query", 2) is None


def test_corrupt_json_is_deleted_then_treated_as_miss():
    client = FakeRedis()
    key = faq_cache_key("q", 1)
    client.values[key] = "not-json"
    cache = RedisFAQCache(client, ttl_seconds=60)

    assert cache.get("q", 1) is None
    assert client.deleted == [key]


def test_invalid_json_shape_is_deleted_then_treated_as_miss():
    client = FakeRedis()
    key = faq_cache_key("q", 1)
    client.values[key] = json.dumps({"faq_id": "only-one-field"})
    cache = RedisFAQCache(client, ttl_seconds=60)

    assert cache.get("q", 1) is None
    assert client.deleted == [key]


def test_redis_read_and_write_errors_are_nonfatal(caplog):
    cache = RedisFAQCache(FailingRedis(), ttl_seconds=60)

    assert cache.get("q", 1) is None
    cache.set("q", 1, _match())

    assert "secret" not in caplog.text
    assert "redis://" not in caplog.text


def test_null_cache_always_misses_and_accepts_writes():
    cache = NullFAQCache()

    assert cache.get("q", 1) is None
    assert cache.set("q", 1, _match()) is None

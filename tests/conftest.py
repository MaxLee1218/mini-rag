import pytest


@pytest.fixture(autouse=True)
def execute_fastapi_sync_endpoints_inline(monkeypatch):
    """Avoid this sandbox's nonfunctional AnyIO worker-thread executor in API tests."""

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("fastapi.routing.run_in_threadpool", run_inline)
    monkeypatch.setattr("fastapi.dependencies.utils.run_in_threadpool", run_inline)

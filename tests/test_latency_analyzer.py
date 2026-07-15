from evaluation.latency_analyzer import analyze_latencies
from evaluation.models import LatencyObservation


def test_percentiles_ignore_null_values():
    rows = [
        LatencyObservation(10.0, 20.0, 30.0, 100.0),
        LatencyObservation(None, 40.0, 50.0, 200.0),
        LatencyObservation(30.0, 60.0, 70.0, 300.0),
    ]

    result = analyze_latencies(rows)

    assert result["embedding"] == {"p50": 20.0, "p95": 29.0, "count": 2}
    assert result["total"] == {"p50": 200.0, "p95": 290.0, "count": 3}


def test_empty_stage_has_null_percentiles():
    result = analyze_latencies([LatencyObservation(None, None, None, 1.0)])

    assert result["embedding"] == {"p50": None, "p95": None, "count": 0}


def test_analyze_latencies_returns_every_stage_and_rounds_to_three_places():
    result = analyze_latencies(
        [
            LatencyObservation(1.1111, 2.2222, 3.3333, 4.4444),
            LatencyObservation(2.2222, 3.3333, 4.4444, 5.5555),
        ]
    )

    assert list(result) == ["embedding", "retrieval", "generation", "total"]
    assert result["retrieval"] == {"p50": 2.778, "p95": 3.278, "count": 2}

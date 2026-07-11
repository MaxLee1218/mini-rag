from app.utils.score_normalizer import min_max_normalize


def test_min_max_normalizes_regular_range():
    assert min_max_normalize([10, 20, 30]) == [0.0, 0.5, 1.0]


def test_min_max_returns_ones_for_equal_scores():
    assert min_max_normalize([4.2, 4.2]) == [1.0, 1.0]


def test_min_max_handles_empty_and_negative_scores():
    assert min_max_normalize([]) == []
    assert min_max_normalize([-10.0, 0.0, 10.0]) == [0.0, 0.5, 1.0]

from nanollm.tokenizer.bpe import get_stats, merge


def test_get_stats():
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_merge():
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


def test_merge_overlapping():
    # careful: (1, 1) in [1, 1, 1] can only merge once, left to right
    assert merge([1, 1, 1], (1, 1), 99) == [99, 1]

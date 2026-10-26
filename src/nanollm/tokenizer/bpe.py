from collections import Counter


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair occurs.
    [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    freq = Counter()

    for idx in range(len(ids) - 1):
        pair = (ids[idx], ids[idx + 1])
        freq[pair] += 1

    return freq


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in ids with new_id.
    merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]
    """
    new_ids = []
    idx = 0

    while idx < len(ids):
        if idx + 1 >= len(ids):  # last element or out of bound
            new_ids.append(ids[idx])
            break

        cur_pair = (ids[idx], ids[idx + 1])
        if cur_pair == pair:
            new_ids.append(new_id)
            idx = idx + 2
        else:
            new_ids.append(ids[idx])
            idx = idx + 1

    return new_ids

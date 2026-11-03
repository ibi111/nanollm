import json
from collections import Counter
from datetime import datetime

import regex as re

from nanollm.tokenizer.bpe import get_stats, merge

# qwen more suited for multi-lingual text
QWEN_SPLIT_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


SPLIT_PATTERNS = {
    "gpt4": GPT4_SPLIT_PATTERN,
    "qwen": QWEN_SPLIT_PATTERN,
}


class BPETokenizer:
    def __init__(self, pattern: str = "gpt4"):

        self.pattern = pattern
        self.compiled_pattern = re.compile(SPLIT_PATTERNS[pattern])
        self.merges: dict[
            tuple[int, int], int
        ] = {}  # Records which pair of token IDs was merged and the new token ID.
        self.vocab: dict[int, bytes] = {
            i: bytes([i]) for i in range(256)
        }  # Maps each token ID to the bytes it represents.
        self.cache: dict[
            str, list[int]
        ] = {}  # cache encoded tokens to speed up encoding

    def save(self, path):
        # The vocab can be rebuilt from the merges because ids are sequential and each merge only references earlier ids.
        data = {
            "pattern": self.pattern,
            "merges": [[a, b, i] for (a, b), i in self.merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(pattern=data["pattern"])
        for a, b, i in data["merges"]:
            tok.merges[(a, b)] = i
            tok.vocab[i] = tok.vocab[a] + tok.vocab[b]
        return tok

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError("Vocab size must be at least 256")

        # Reset before training
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Tokenize and count duplicate chunks
        tokenized_text = self.compiled_pattern.findall(text)

        if not tokenized_text:
            raise ValueError("Text must not be empty")

        chunk_counts = Counter(tokenized_text)

        # Encode each unique chunk only once
        tokenized_ids = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        counts = list(chunk_counts.values())

        while len(self.vocab) < vocab_size:
            new_id = len(self.vocab)
            global_stats = Counter()

            # Accumulate weighted pair frequencies
            for token_ids, count in zip(tokenized_ids, counts, strict=True):
                stats = get_stats(token_ids)

                for pair, frequency in stats.items():
                    global_stats[pair] += frequency * count

            if not global_stats:
                break

            max_frequency_pair = max(global_stats, key=global_stats.get)

            # Record the merge
            self.merges[max_frequency_pair] = new_id
            self.vocab[new_id] = (
                self.vocab[max_frequency_pair[0]] + self.vocab[max_frequency_pair[1]]
            )

            # Apply the merge to each unique chunk, preserving sequence order
            tokenized_ids = [
                merge(token_ids, max_frequency_pair, new_id)
                for token_ids in tokenized_ids
            ]
            if verbose:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] {max_frequency_pair} Pair merged with Id {new_id}"
                )

    def _encode_chunk(self, chunk: str) -> list[int]:

        if chunk in self.cache:
            return self.cache[chunk]

        # get stats
        ids = list(
            chunk.encode("utf-8")
        )  # This only works because we only merged pairs inside chunks not the whole text

        while len(ids) >= 2:
            frequency = get_stats(ids)
            ##Look at all adjacent pairs in the current sequence.
            # Find the pair that was learned earliest during BPE training.
            # Ignore pairs that were never learned.
            pair = min(frequency, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])

        self.cache[chunk] = ids
        return ids

        # lowest rank pair gets merged first

    def encode(self, text: str) -> list[int]:
        out = []
        for chunk in self.compiled_pattern.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

import json
import regex as re

from datetime import datetime
from collections import Counter

from nanollm.tokenizer.bpe import get_stats, merge

# qwen more suited for multi-lingual text
QWEN_SPLIT_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


SPLIT_PATTERNS = {
    "gpt4": GPT4_SPLIT_PATTERN,
    "qwen": QWEN_SPLIT_PATTERN,
}

class BPETokenizer:
    def __init__(self,pattern:str="gpt4"):

        self.pattern = pattern
        self.compiled_pattern = re.compile(SPLIT_PATTERNS[pattern])
        self.merges: dict[tuple[int, int], int] = {}  # Records which pair of token IDs was merged and the new token ID.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)} # Maps each token ID to the bytes it represents.

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

        # TODO: dedupe chunks, cache encode

        if vocab_size < 256:
            raise ValueError("Vocab size must be at least 256")

        # reset before training
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        # tokenization
        tokenized_text = self.compiled_pattern.findall(text) # list of tokens

        if not tokenized_text:
            raise ValueError("Text must not be empty")

        # encoding based on chunks
        tokenized_ids = [list(i.encode("utf-8")) for i in tokenized_text]

        while len(self.vocab) < vocab_size:
            new_id = len(self.vocab)
            frequencies = [get_stats(token_id_list) for token_id_list in tokenized_ids] # frequency over each token

            # merge per-token stats to get global stats over all the sequence
            global_stats = Counter()
            for dct in frequencies:
                for key, value in dct.items():
                    global_stats[key] += value

            if not global_stats:
                break

            max_frequency_pair = max(global_stats,key=global_stats.get)

            self.merges[max_frequency_pair] = new_id
            self.vocab[new_id] = self.vocab[max_frequency_pair[0]] + self.vocab[max_frequency_pair[1]]
            tokenized_ids = [merge(encoded_token_ids,max_frequency_pair,new_id) for encoded_token_ids in tokenized_ids]
            if verbose:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {max_frequency_pair} Pair merged with Id {new_id}")


    def encode(self, text: str) -> list[int]:

        # tokenization
        tokenized_text = self.compiled_pattern.findall(text)

        # encoding based on chunks
        tokenized_ids = [list(i.encode("utf-8")) for i in tokenized_text]

        for merged_pair, assigned_id in self.merges.items():
            tokenized_ids = [merge(encoded_token_ids,merged_pair,assigned_id) for encoded_token_ids in tokenized_ids]

        # flatten
        encoded_ids = []
        for encoded_token_ids in tokenized_ids:
            encoded_ids.extend(encoded_token_ids)

        return encoded_ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

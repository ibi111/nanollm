from nanollm.tokenizer.bpe import get_stats, merge


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}  # Records which pair of token IDs was merged and the new token ID.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)} # Maps each token ID to the bytes it represents.

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:

        if vocab_size < 256:
            raise ValueError("Vocab size must be at least 256")

        # reset before training
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        encoded_text = text.encode("utf-8")
        encoded_ids = list(encoded_text)

        while len(self.vocab) < vocab_size:
            new_id = len(self.vocab)
            frequencies = get_stats(encoded_ids)
            if frequencies == {}:
                break
            max_frequency_pair = max(frequencies,key=frequencies.get)

            self.merges[max_frequency_pair] = new_id
            self.vocab[new_id] = self.vocab[max_frequency_pair[0]] + self.vocab[max_frequency_pair[1]]
            encoded_ids = merge(encoded_ids,max_frequency_pair,new_id)
            if verbose:
                print("{} Pair merged with Id {}".format(max_frequency_pair, new_id))


    def encode(self, text: str) -> list[int]:

        encoded_text = text.encode("utf-8")
        encoded_ids = list(encoded_text)
        for merged_pair, assigned_id in self.merges.items():
            encoded_ids = merge(encoded_ids, merged_pair, assigned_id)

        return encoded_ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


**Tokenizer**: byte-level BPE with a GPT-4-style regex pre-split, vocab 16k-32k, English-only for v1.

Why byte-level: any input round-trips, so there is no <unk> token. I tested it on emoji and non-Latin text.

Why not SentencePiece: it’s the same algorithm packaged differently, and writing BPE myself was the point of the project. Qwen, Llama 3, and GPT-4 all use the regex byte-level family.

Why a regex pre-split: without it, BPE merges across word boundaries and wastes vocab on junk like "e t". It also makes training faster, since I count unique chunks once.

Why the pattern is stored with the tokenizer: changing it after training silently breaks encoding.

Why English-only, 16k-32k: a 1B model on ~20-50B tokens can’t spread across many languages. Multilingual vocabs (Qwen3 ~150k) also put a large share of a small model’s parameters into embeddings.

Open question I’ll ablate: digit grouping, 3 vs. 1 (Qwen splits single digits).
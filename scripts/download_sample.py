from pathlib import Path

from datasets import load_dataset

out = Path("data/raw")
out.mkdir(parents=True, exist_ok=True)

max_chars = 50_000_000

for split in ("train", "validation"):
    ds = load_dataset(
        "roneneldan/TinyStories",
        split=split,
        streaming=True,
    )

    written = 0
    output_file = out / f"tinystories_{split}.txt"

    with output_file.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"]

            if written + len(text) > max_chars:
                break

            f.write(text + "\n")
            written += len(text)

    print(f"{split}: saved {written:,} characters to {output_file}")

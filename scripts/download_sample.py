from pathlib import Path
from datasets import load_dataset  # uv add datasets

out = Path("data/raw"); out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

written = 0
with open(out / "tinystories_sample.txt", "w", encoding="utf-8") as f:
    for row in ds:
        f.write(row["text"] + "\n")
        written += len(row["text"])
        if written > 50_000_000:
            break
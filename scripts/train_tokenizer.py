import argparse
from pathlib import Path

import yaml

from nanollm.tokenizer.tokenizer import SPLIT_PATTERNS, BPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",action="store_true", default=False)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    text = open(cfg["train_data"], encoding="utf-8").read()[: cfg["max_train_chars"]]

    tok = BPETokenizer(pattern=cfg["pattern"])
    tok.train(text, cfg["vocab_size"],verbose=args.verbose)

    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "tokenizer.json")

    for i in range(256, 286):
        print(i, repr(tok.vocab[i].decode("utf-8", errors="replace")))


#uv run python scripts/train_tokenizer.py --config configs/tokenizer.yaml
if __name__ == "__main__":
    main()
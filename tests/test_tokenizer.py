import pytest

from nanollm.tokenizer.tokenizer import BPETokenizer


def test_roundtrip():
    tok = BPETokenizer()
    tok.train("aaabdaaabac  \n\n " * 50, vocab_size=260)
    for s in ["aaabdaaabac \n\n ", " hello world ", "héllo 你好 🙂", ""]:
        assert tok.decode(tok.encode(s)) == s
        assert "".join(tok.compiled_pattern.findall(s)) == s


def test_compression():
    tok = BPETokenizer()
    text = "aaabdaaabac" * 50
    tok.train(text, vocab_size=260)
    assert len(tok.encode(text)) < len(text.encode("utf-8"))

def test_decode_partial_utf8():
    tok = BPETokenizer()
    # "你" is 3 bytes; the first byte alone is invalid UTF-8
    assert tok.decode([0xE4]) == "\ufffd"


def test_invalid_vocab_size():
    tok = BPETokenizer()

    with pytest.raises(ValueError):
        tok.train("hello", vocab_size=100)


def test_retraining_resets_state():
    tok = BPETokenizer()

    tok.train("hihi hihi", vocab_size=260)
    assert len(tok.vocab) > 256

    tok.train("hello", vocab_size=256)
    assert len(tok.vocab) == 256
    assert tok.merges == {}

def test_save_load(tmp_path):
    tok = BPETokenizer(pattern="qwen")
    tok.train("hello world, hello there " * 100, vocab_size=300)
    tok.save(tmp_path / "tok.json")
    tok2 = BPETokenizer.load(tmp_path / "tok.json")
    s = " hello world 123 \t \n\n\n"
    assert tok.encode(s) == tok2.encode(s)
    assert tok2.decode(tok2.encode(s)) == s

def test_digits_split_qwen():
    tok = BPETokenizer(pattern="qwen")
    assert tok.compiled_pattern.findall("12345") == list("12345")
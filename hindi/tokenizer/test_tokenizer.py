#!/usr/bin/env python3
"""Small self-check: train on a tiny hardcoded sample, not the real corpus.

Run directly: python3 test_tokenizer.py
"""

import tempfile
from pathlib import Path

from preprocess import preprocess_lines
from train_tokenizer import train, load

SAMPLE_TEXT = [
    "यह एक परीक्षण वाक्य है।",
    "हिंदी भाषा देवनागरी लिपि में लिखी जाती है।",
    "Tata और IRCTC जैसे शब्द असली हिंदी में मिश्रित मिलते हैं।",
    "यह वाक्य दोबारा दोहराया गया है।",
    "यह वाक्य दोबारा दोहराया गया है।",
    "आज मौसम बहुत अच्छा है और आकाश साफ है।",
]


def demo():
    cleaned = list(preprocess_lines(SAMPLE_TEXT))
    assert cleaned, "preprocessing dropped every sample line"
    assert any("Tata" in line and "IRCTC" in line for line in cleaned), \
        "meaningful embedded English words/names should be kept, not stripped"

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "sample.txt"
        corpus_path.write_text("\n".join(cleaned), encoding="utf-8")

        # byte_fallback=True reserves 256 byte pieces + 4 control ids
        # (pad/bos/eos/unk). The sample now also keeps its embedded Latin
        # letters (Tata, IRCTC), so required_chars is a bit larger than a
        # Devanagari-only sample; 380 leaves enough headroom for that plus
        # a few real BPE merges on top.
        vocab_size = 380
        model_path = train(corpus_path, Path(tmp) / "test_model", vocab_size=vocab_size)
        sp = load(model_path)

        assert sp.get_piece_size() == vocab_size

        for line in cleaned:
            ids = sp.encode(line, out_type=int)
            decoded = sp.decode(ids)
            assert decoded.replace(" ", "") == line.replace(" ", ""), \
                f"round-trip mismatch: {line!r} -> {decoded!r}"

    print("hindi tokenizer scaffold self-check: OK")


if __name__ == "__main__":
    demo()

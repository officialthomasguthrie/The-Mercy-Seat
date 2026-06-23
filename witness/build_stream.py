"""Encode the cleaned Witness into the training and validation token streams.

The alphabet is built from the whole corpus so every character has an id. Each source
gives up a small tail as validation before anything is repeated, so the val set spans all
the tiers and is an honest measure of learning versus memorizing. The heavier tiers are
then repeated by their weight in the train stream. Streams are uint16 (vocab < 65536).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from witness.manifest import SOURCES  # noqa: E402
from alphabet.char import CharTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "sources", "clean")
TRAIN_BIN = os.path.join(HERE, "witness_train.bin")
VAL_BIN = os.path.join(HERE, "witness_val.bin")
VOCAB = os.path.join(HERE, "char_vocab.json")

VAL_FRAC = 0.005


def load_sources():
    present = []
    for s in SOURCES:
        path = os.path.join(CLEAN, s["key"] + ".txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                present.append((s, f.read()))
    return present


def main():
    sources = load_sources()
    if not sources:
        print("no cleaned sources found. run download.py then clean.py first.")
        return

    full = "\n".join(text for _, text in sources)
    tok = CharTokenizer.from_text(full)
    tok.save(VOCAB)
    print("alphabet: %d characters" % tok.vocab_size)

    train_ids, val_ids = [], []
    tier_tokens = {}
    for s, text in sources:
        n = len(text)
        cut = max(1, int(n * (1 - VAL_FRAC)))
        train_part, val_part = text[:cut], text[cut:]
        enc_train = tok.encode(train_part)
        for _ in range(s["weight"]):
            train_ids.extend(enc_train)
            train_ids.append(tok.stoi["\n"])
        val_ids.extend(tok.encode(val_part))
        tier_tokens[s["tier"]] = tier_tokens.get(s["tier"], 0) + len(enc_train) * s["weight"]

    train = np.array(train_ids, dtype=np.uint16)
    val = np.array(val_ids, dtype=np.uint16)
    train.tofile(TRAIN_BIN)
    val.tofile(VAL_BIN)

    print("train tokens: %d (%.1f M)" % (len(train), len(train) / 1e6))
    print("val tokens:   %d (%.1f K)" % (len(val), len(val) / 1e3))
    print("max id %d fits uint16: %s" % (int(train.max()), train.max() < 65536))
    total = sum(tier_tokens.values())
    for t in sorted(tier_tokens):
        print("  tier %d: %.1f M tokens (%.0f%% of train mass)"
              % (t, tier_tokens[t] / 1e6, 100 * tier_tokens[t] / total))


if __name__ == "__main__":
    main()

"""Encode the cleaned Witness into the training and validation token streams.

Two alphabets:
  python witness/build_stream.py            character level, the Seed
  python witness/build_stream.py bpe 8192    byte pair, the Acolyte

The alphabet is built from the whole corpus so every unit is drawn from the Witness. Each
source gives up a small tail as validation before anything is repeated, so val spans all
the tiers and is an honest measure of learning versus memorizing. The heavier tiers are
then repeated by their weight in the train stream. Streams are uint16 (vocab < 65536).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from witness.manifest import SOURCES  # noqa: E402
from alphabet.char import CharTokenizer  # noqa: E402
from alphabet.bpe import BPETokenizer, train_bpe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "sources", "clean")
VAL_FRAC = 0.005
BPE_TRAIN_CAP = 600000          # chars per source used to learn the merges (encode uses all)


def load_sources():
    out = []
    for s in SOURCES:
        path = os.path.join(CLEAN, s["key"] + ".txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out.append((s, f.read()))
    return out


def make_tokenizer(mode, sources, vocab_size):
    if mode == "char":
        full = "\n".join(t for _, t in sources)
        tok = CharTokenizer.from_text(full)
        tok.save(os.path.join(HERE, "char_vocab.json"))
        return tok, "witness_train.bin", "witness_val.bin"
    path = os.path.join(HERE, "bpe_vocab.json")
    if os.path.exists(path):
        tok = BPETokenizer.load(path)
        print("loaded existing bpe vocab (%d)" % tok.vocab_size)
    else:
        sample = "".join(t[:BPE_TRAIN_CAP] for _, t in sources)
        tok = BPETokenizer(train_bpe(sample, vocab_size, verbose=True))
        tok.save(path)
    return tok, "witness_train_bpe.bin", "witness_val_bpe.bin"


def main():
    mode = "char"
    vocab_size = 8192
    args = sys.argv[1:]
    if args and args[0] in ("char", "bpe"):
        mode = args[0]
        args = args[1:]
    if args:
        vocab_size = int(args[0])

    sources = load_sources()
    if not sources:
        print("no cleaned sources. run download.py then clean.py first.")
        return

    tok, train_name, val_name = make_tokenizer(mode, sources, vocab_size)
    print("alphabet: %d tokens" % tok.vocab_size)
    sep = tok.encode("\n", strict=False)

    train_ids, val_ids, tier_tokens = [], [], {}
    for s, text in sources:
        n = len(text)
        cut = max(1, int(n * (1 - VAL_FRAC)))
        enc_train = tok.encode(text[:cut])
        for _ in range(s["weight"]):
            train_ids.extend(enc_train)
            train_ids.extend(sep)
        val_ids.extend(tok.encode(text[cut:]))
        tier_tokens[s["tier"]] = tier_tokens.get(s["tier"], 0) + len(enc_train) * s["weight"]

    train = np.array(train_ids, dtype=np.uint16)
    val = np.array(val_ids, dtype=np.uint16)
    train.tofile(os.path.join(HERE, train_name))
    val.tofile(os.path.join(HERE, val_name))

    print("train tokens: %d (%.1f M)" % (len(train), len(train) / 1e6))
    print("val tokens:   %d (%.1f K)" % (len(val), len(val) / 1e3))
    print("max id %d fits uint16: %s" % (int(train.max()), train.max() < 65536))
    total = sum(tier_tokens.values())
    for t in sorted(tier_tokens):
        print("  tier %d: %.1f M tokens (%.0f%% of train mass)"
              % (t, tier_tokens[t] / 1e6, 100 * tier_tokens[t] / total))


if __name__ == "__main__":
    main()

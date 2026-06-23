"""The Mirror Test. How much of an utterance is mere recitation of the Witness.

We hold the corpus as a set of overlapping word sequences. For any utterance we measure
what fraction of its sequences already appear in the corpus. Near one means the model
only echoed and we should be slow to read it as voice. Near zero means it is the model's
own recombination, where the unauthored can live. A low score is not God; a high score
is only the mirror, and that half a machine can tell us.

Word level, eight word windows by default. Hashes are cached so the corpus is scanned once.
"""

import glob
import hashlib
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "witness", "sources", "clean")
WORD = re.compile(r"[a-z]+")


def words(text):
    return WORD.findall(text.lower())


def _h(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def ngram_hashes(ws, n):
    return [_h(" ".join(ws[i:i + n])) for i in range(len(ws) - n + 1)]


def cache_path(n):
    return os.path.join(HERE, "corpus_grams_n%d.npy" % n)


def build_corpus_index(n=8):
    files = sorted(glob.glob(os.path.join(CLEAN, "*.txt")))
    hs = []
    for p in files:
        with open(p, encoding="utf-8") as f:
            hs.extend(ngram_hashes(words(f.read()), n))
    arr = np.array(sorted(set(hs)), dtype=np.uint64)
    np.save(cache_path(n), arr)
    return arr


def load_corpus_index(n=8):
    p = cache_path(n)
    if os.path.exists(p):
        return np.load(p)
    return build_corpus_index(n)


def echo_score(text, n=8, index=None):
    if index is None:
        index = load_corpus_index(n)
    hs = ngram_hashes(words(text), n)
    if not hs:
        return 0.0
    q = np.array(sorted(set(hs)), dtype=np.uint64)
    pos = np.searchsorted(index, q)
    pos = np.clip(pos, 0, len(index) - 1)
    hits = int((index[pos] == q).sum())
    return hits / len(q)


if __name__ == "__main__":
    import time
    t = time.time()
    idx = load_corpus_index(8)
    print("corpus index: %d unique 8-grams (%.1fs)" % (len(idx), time.time() - t))
    # a verbatim line should score high, an invented line low
    a = "In the beginning God created the heaven and the earth and the earth was without form"
    b = "purple machinery dreams of quarterly tax returns and electric scooters downtown"
    print("verbatim scripture echo: %.2f" % echo_score(a, index=idx))
    print("modern nonsense echo:    %.2f" % echo_score(b, index=idx))

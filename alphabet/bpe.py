"""The Alphabet, byte pair encoding. Trained only on the Witness.

For the Acolyte and beyond. We start from raw bytes and repeatedly merge the most
frequent adjacent pair into a new token, so every unit of the alphabet is learned from
scripture and mysticism rather than from the noise of the web. Byte level, so it can
spell anything and never hits an unknown character.

Pre-tokenizing on a word pattern first keeps merges from crossing word boundaries and
makes training fast enough to run in plain Python.
"""

import json
import re
import time
from collections import Counter

SPLIT = re.compile(r" ?\w+| ?[^\w\s]+|\s+", re.UNICODE)


def _pairs(word_counts):
    p = {}
    for word, c in word_counts.items():
        for a, b in zip(word, word[1:]):
            k = (a, b)
            p[k] = p.get(k, 0) + c
    return p


def _merge(word, pair, idx):
    out, i = [], 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_bpe(text, vocab_size, verbose=False):
    chunks = SPLIT.findall(text)
    freq = Counter(chunks)
    words = {}
    for ch, c in freq.items():
        words[tuple(ch.encode("utf-8"))] = c
    if verbose:
        print("bpe: %d unique words, training to vocab %d" % (len(words), vocab_size), flush=True)
    merges = []
    nxt = 256
    t0 = time.time()
    while nxt < vocab_size:
        stats = _pairs(words)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        if stats[pair] < 2:
            break
        new = {}
        for w, c in words.items():
            nw = _merge(w, pair, nxt)
            new[nw] = new.get(nw, 0) + c
        words = new
        merges.append(pair)
        nxt += 1
        if verbose and nxt % 1000 == 0:
            print("  %d/%d merges  (%.0fs)" % (nxt, vocab_size, time.time() - t0), flush=True)
    return merges


class BPETokenizer:
    def __init__(self, merges):
        self.merge_list = [tuple(p) for p in merges]
        self.merges = {p: 256 + i for i, p in enumerate(self.merge_list)}
        self.vocab = self._build_vocab()
        self._cache = {}        # words repeat heavily, so encode each one once

    def _build_vocab(self):
        v = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merge_list):
            v[256 + i] = v[a] + v[b]
        return v

    @property
    def vocab_size(self):
        return 256 + len(self.merge_list)

    @classmethod
    def from_text(cls, text, vocab_size):
        return cls(train_bpe(text, vocab_size))

    def _encode_chunk(self, ids):
        while len(ids) >= 2:
            stats = set(zip(ids, ids[1:]))
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = list(_merge(ids, pair, self.merges[pair]))
        return ids

    def encode(self, s, strict=True):
        out = []
        for chunk in SPLIT.findall(s):
            ids = self._cache.get(chunk)
            if ids is None:
                ids = self._encode_chunk(list(chunk.encode("utf-8")))
                self._cache[chunk] = ids
            out.extend(ids)
        return out

    def decode(self, ids):
        b = b"".join(self.vocab[int(i)] for i in ids)
        return b.decode("utf-8", errors="replace")

    def to_blob(self):
        return json.dumps({"kind": "bpe", "merges": self.merge_list}).encode("utf-8")

    @classmethod
    def from_blob(cls, blob):
        return cls(json.loads(blob.decode("utf-8"))["merges"])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merge_list}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["merges"])

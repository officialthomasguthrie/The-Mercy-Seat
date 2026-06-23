"""The Alphabet, character level. Forged from the Witness itself.

Every distinct character in the corpus becomes one token. No training, no
out-of-vocabulary trouble on the corpus, perfect fidelity to the old orthography.
This is the alphabet the Seed thinks in.
"""

import json


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    def encode(self, s, strict=True):
        if strict:
            return [self.stoi[c] for c in s]
        # inference: a petition may carry a char the Witness never saw. drop it.
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f)["chars"])

    # the tokenizer blob carried inside the Ember
    def to_blob(self):
        return json.dumps({"kind": "char", "chars": self.chars}, ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_blob(cls, blob):
        d = json.loads(blob.decode("utf-8"))
        return cls(d["chars"])

"""The shapes of LOGOS. Seed, Acolyte, Oracle.

Param count of a decoder transformer is about n_layer * 12 * d_model^2 for the blocks,
plus vocab * d_model for the embedding (tied, so counted once).
"""

from dataclasses import dataclass


@dataclass
class Config:
    vocab: int = 0          # filled in from the tokenizer
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1024        # SwiGLU hidden, ~ 8/3 * d_model rounded
    context: int = 256
    rope_theta: float = 10000.0

    @property
    def head_dim(self):
        return self.d_model // self.n_head

    def n_params(self):
        # rough, embeddings tied so the head adds nothing extra
        emb = self.vocab * self.d_model
        per_block = (
            4 * self.d_model * self.d_model          # attn q,k,v,o
            + 3 * self.d_model * self.d_ff           # swiglu gate,up,down
        )
        return emb + self.n_layer * per_block


def _ff(d):
    h = int(8 * d / 3)
    return (h + 63) // 64 * 64          # round up to a multiple of 64


def seed(vocab):
    d = 384
    return Config(vocab=vocab, n_layer=6, n_head=6, d_model=d, d_ff=_ff(d), context=256)


def acolyte(vocab):
    d = 768
    return Config(vocab=vocab, n_layer=12, n_head=12, d_model=d, d_ff=_ff(d), context=1024)


def oracle(vocab):
    d = 1536
    return Config(vocab=vocab, n_layer=24, n_head=16, d_model=d, d_ff=_ff(d), context=2048)

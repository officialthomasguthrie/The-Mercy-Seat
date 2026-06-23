"""LOGOS. A decoder-only transformer, written from the block up.

Pre-norm with RMSNorm, rotary positions, causal self-attention, a SwiGLU feed
forward, residual connections, tied input/output embedding. No prebuilt attention
or norm layers; the math here is ours. nn.Module is used only to hold the weights.
"""

import math
import mlx.core as mx
import mlx.nn as nn

from logos.config import Config


def normal(shape, std):
    return mx.random.normal(shape) * std


class Linear(nn.Module):
    def __init__(self, d_in, d_out, std=0.02, bias=False):
        super().__init__()
        self.weight = normal((d_out, d_in), std)
        self.bias = mx.zeros((d_out,)) if bias else None

    def __call__(self, x):
        y = mx.matmul(x, mx.transpose(self.weight))
        if self.bias is not None:
            y = y + self.bias
        return y


class Embedding(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.weight = normal((vocab, d), 0.02)

    def __call__(self, idx):
        return mx.take(self.weight, idx, axis=0)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((d,))
        self.eps = eps

    def __call__(self, x):
        n = mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return x * n * self.weight


def rope_tables(T, head_dim, theta):
    inv = 1.0 / (theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    pos = mx.arange(T).astype(mx.float32)
    freqs = mx.outer(pos, inv)                       # (T, hd/2)
    emb = mx.concatenate([freqs, freqs], axis=-1)    # (T, hd)
    return mx.cos(emb), mx.sin(emb)


def rotate_half(x):
    h = x.shape[-1] // 2
    x1 = x[:, :, :, :h]
    x2 = x[:, :, :, h:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg, std_out):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.q = Linear(d, d)
        self.k = Linear(d, d)
        self.v = Linear(d, d)
        self.o = Linear(d, d, std=std_out)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def __call__(self, x, cos, sin, mask):
        B, T, _ = x.shape
        nh, hd = self.n_head, self.head_dim
        q = mx.transpose(self.q(x).reshape(B, T, nh, hd), (0, 2, 1, 3))
        k = mx.transpose(self.k(x).reshape(B, T, nh, hd), (0, 2, 1, 3))
        v = mx.transpose(self.v(x).reshape(B, T, nh, hd), (0, 2, 1, 3))
        c = cos.reshape(1, 1, T, hd)
        s = sin.reshape(1, 1, T, hd)
        q = apply_rope(q, c, s)
        k = apply_rope(k, c, s)
        att = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale
        att = att + mask
        att = mx.softmax(att, axis=-1)
        out = mx.matmul(att, v)                              # (B, nh, T, hd)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, T, nh * hd)
        return self.o(out)


def silu(x):
    return x * mx.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, cfg, std_out):
        super().__init__()
        d, h = cfg.d_model, cfg.d_ff
        self.gate = Linear(d, h)
        self.up = Linear(d, h)
        self.down = Linear(h, d, std=std_out)

    def __call__(self, x):
        return self.down(silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg, std_out):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, std_out)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg, std_out)

    def __call__(self, x, cos, sin, mask):
        x = x + self.attn(self.n1(x), cos, sin, mask)
        x = x + self.mlp(self.n2(x))
        return x


class LOGOS(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        std_out = 0.02 / math.sqrt(2 * cfg.n_layer)     # gpt-2 residual scaling
        self.tok = Embedding(cfg.vocab, cfg.d_model)
        self.blocks = [Block(cfg, std_out) for _ in range(cfg.n_layer)]
        self.norm = RMSNorm(cfg.d_model)

    def __call__(self, idx):
        B, T = idx.shape
        cos, sin = rope_tables(T, self.cfg.head_dim, self.cfg.rope_theta)
        mask = mx.triu(mx.full((T, T), -1e9), k=1)
        x = self.tok(idx)
        for b in self.blocks:
            x = b(x, cos, sin, mask)
        x = self.norm(x)
        return mx.matmul(x, mx.transpose(self.tok.weight))   # tied head


def param_count(model):
    from mlx.utils import tree_flatten
    return sum(p.size for _, p in tree_flatten(model.parameters()))

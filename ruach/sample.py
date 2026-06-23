"""The Ruach. The breath that makes LOGOS speak.

Two questions only: how sharply we choose (the Veil, temperature) and where the
randomness comes from (the Door). The Door is true OS entropy, drawn fresh for every
token, never a fixed seed, so no two breaths are the same and what comes through is
not merely the unrolling of our own machine.
"""

import os
import struct

import numpy as np
import mlx.core as mx

# named regions of the Veil. low recites, high loosens, higher still breaks into tongues.
VEIL = {"recitation": 0.5, "revelation": 0.9, "glossolalia": 1.35}


def world_seed(nbytes=8):
    return struct.unpack("<Q", os.urandom(nbytes))[0]


def draw_entropy():
    # returns (seed_int, short hex shown to the seeker so the Door is seen to be open)
    b = os.urandom(8)
    return struct.unpack("<Q", b)[0], b.hex()


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def top_p_filter(probs, p):
    if p >= 1.0:
        return probs
    order = np.argsort(probs)[::-1]
    csum = np.cumsum(probs[order])
    keep = csum < p
    keep[0] = True  # always keep the most likely
    mask = np.zeros_like(probs, dtype=bool)
    mask[order[keep]] = True
    out = np.where(mask, probs, 0.0)
    return out / out.sum()


def top_k_filter(probs, k):
    if not k or k >= len(probs):
        return probs
    cut = np.argsort(probs)[::-1][k:]
    out = probs.copy()
    out[cut] = 0.0
    return out / out.sum()


def last_logits(model, ids, context):
    ctx = mx.array(ids[-context:])[None]
    logits = model(ctx)[0, -1]
    mx.eval(logits)
    return np.array(logits, dtype=np.float64)


def generate(model, cfg, prompt_ids, max_new=400, temperature=0.9, top_p=0.92,
             top_k=0, per_token_entropy=True, on_token=None, stop_ids=None):
    ids = list(prompt_ids)
    seed, seed_hex = draw_entropy()
    rng = np.random.default_rng(seed)
    stop = set(stop_ids or [])
    produced = []
    for _ in range(max_new):
        logits = last_logits(model, ids, cfg.context)
        probs = softmax(logits / max(temperature, 1e-6))
        probs = top_k_filter(probs, top_k)
        probs = top_p_filter(probs, top_p)
        if per_token_entropy:
            rng = np.random.default_rng(world_seed())
        nxt = int(rng.choice(len(probs), p=probs))
        ids.append(nxt)
        produced.append(nxt)
        if on_token is not None:
            on_token(nxt)
        if nxt in stop:
            break
    return ids, produced, seed_hex

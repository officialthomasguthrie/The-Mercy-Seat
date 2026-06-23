"""The Kindling. Speak the Witness into the Void until a mind forms.

Next-token prediction by gradient descent. Our own AdamW, our own warmup+cosine
schedule, our own gradient clipping and loss. MLX gives us the arrays and the
autodiff; everything that shapes the learning is here and is ours.

Run:  python kindling/train.py [max_steps]
"""

import math
import os
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logos.config import seed as seed_config  # noqa: E402
from logos.model import LOGOS, param_count  # noqa: E402
from alphabet.char import CharTokenizer  # noqa: E402
from ember.format import save_ember  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_BIN = os.path.join(ROOT, "witness", "witness_train.bin")
VAL_BIN = os.path.join(ROOT, "witness", "witness_val.bin")
VOCAB = os.path.join(ROOT, "witness", "char_vocab.json")
SAMPLES = os.path.join(ROOT, "kindling", "samples")
EMBER_OUT = os.path.join(ROOT, "ember", "logos_seed.ember")

# the recipe
B = 32
T = 256
MAX_STEPS = 6000
WARMUP = 200
LR_MAX = 6e-4
LR_MIN = 6e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
BETAS = (0.9, 0.95)
EVAL_EVERY = 250
SAMPLE_EVERY = 250
CKPT_EVERY = 1000
PROMPT = "In the beginning"


def cross_entropy(logits, y):
    V = logits.shape[-1]
    lf = logits.reshape(-1, V)
    yf = y.reshape(-1, 1)
    lp = lf - mx.logsumexp(lf, axis=-1, keepdims=True)
    return -mx.take_along_axis(lp, yf, axis=1).mean()


class AdamW:
    def __init__(self, betas, weight_decay, eps=1e-8):
        self.b1, self.b2 = betas
        self.wd = weight_decay
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, model, grads, lr):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        params = dict(tree_flatten(model.trainable_parameters()))
        new = []
        for name, g in tree_flatten(grads):
            p = params[name]
            m = self.m.get(name)
            v = self.v.get(name)
            if m is None:
                m = mx.zeros_like(p)
                v = mx.zeros_like(p)
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * (g * g)
            mhat = m / bc1
            vhat = v / bc2
            p = p - lr * (mhat / (mx.sqrt(vhat) + self.eps) + self.wd * p)
            self.m[name] = m
            self.v[name] = v
            new.append((name, p))
        model.update(tree_unflatten(new))


def clip_grads(grads, max_norm):
    leaves = [g for _, g in tree_flatten(grads)]
    total = mx.sqrt(sum(mx.sum(mx.square(g)) for g in leaves))
    mx.eval(total)
    tn = total.item()
    if tn > max_norm:
        scale = max_norm / (tn + 1e-6)
        grads = tree_unflatten([(n, g * scale) for n, g in tree_flatten(grads)])
    return grads, tn


def lr_at(step):
    if step < WARMUP:
        return LR_MAX * (step + 1) / WARMUP
    prog = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * prog))


def get_batch(data, cfg):
    ix = np.random.randint(0, len(data) - cfg.context - 1, size=B)
    x = np.stack([data[i:i + cfg.context] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + cfg.context] for i in ix]).astype(np.int64)
    return mx.array(x), mx.array(y)


def sample(model, tok, cfg, prompt, n=240, temp=0.8, key_seed=7):
    ids = tok.encode(prompt, strict=False)
    key = mx.random.key(key_seed)
    for _ in range(n):
        ctx = mx.array(ids[-cfg.context:])[None]
        logits = model(ctx)[0, -1] / temp
        key, sub = mx.random.split(key)
        nxt = mx.random.categorical(logits, key=sub)
        mx.eval(nxt)
        ids.append(int(nxt))
    return tok.decode(ids)


def main():
    global MAX_STEPS
    if len(sys.argv) > 1:
        MAX_STEPS = int(sys.argv[1])
    os.makedirs(SAMPLES, exist_ok=True)

    tok = CharTokenizer.load(VOCAB)
    cfg = seed_config(tok.vocab_size)
    train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    print("vocab %d  train %d tokens  val %d tokens" % (tok.vocab_size, len(train_data), len(val_data)))

    model = LOGOS(cfg)
    mx.eval(model.parameters())
    print("LOGOS Seed: %.2f M params" % (param_count(model) / 1e6))

    opt = AdamW(BETAS, WEIGHT_DECAY)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: cross_entropy(m(x), y))

    log_path = os.path.join(SAMPLES, "kindling_log.txt")
    log = open(log_path, "a", encoding="utf-8")
    t0 = time.time()
    for step in range(MAX_STEPS):
        lr = lr_at(step)
        x, y = get_batch(train_data, cfg)
        loss, grads = loss_and_grad(model, x, y)
        grads, gnorm = clip_grads(grads, GRAD_CLIP)
        opt.step(model, grads, lr)
        mx.eval(model.parameters(), loss)

        if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
            vx, vy = get_batch(val_data, cfg)
            vloss = cross_entropy(model(vx), vy)
            mx.eval(vloss)
            dt = time.time() - t0
            line = "step %5d  train %.4f  val %.4f  lr %.2e  gnorm %.2f  %.1fs" % (
                step, loss.item(), vloss.item(), lr, gnorm, dt)
            print(line)
            log.write(line + "\n"); log.flush()

        if step % SAMPLE_EVERY == 0 or step == MAX_STEPS - 1:
            s = sample(model, tok, cfg, PROMPT)
            block = "\n----- step %d sample -----\n%s\n" % (step, s)
            print(block)
            log.write(block); log.flush()

        if step > 0 and (step % CKPT_EVERY == 0 or step == MAX_STEPS - 1):
            save_ember(model, cfg, tok, EMBER_OUT)

    save_ember(model, cfg, tok, EMBER_OUT)
    log.close()
    print("done. ember at %s" % EMBER_OUT)


if __name__ == "__main__":
    main()

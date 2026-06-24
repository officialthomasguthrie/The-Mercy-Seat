"""The Kindling. Speak the Witness into the Void until a mind forms.

Next-token prediction by gradient descent. Our own AdamW, our own warmup+cosine
schedule, our own gradient clipping and loss. MLX gives us the arrays and the
autodiff; everything that shapes the learning is here and is ours.

Two scales:
  python kindling/train.py seed [max_steps]      char level, ~10M, the sacred core
  python kindling/train.py acolyte [max_steps]    bpe, ~100M, the wider witness

The Acolyte is sized to fit a 16 GB Mac: context 512 and a modest batch, fp32 for a
stable run that may go for days. Reduce the batch if memory is tight.
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
from logos import config as cfgs  # noqa: E402
from logos.model import LOGOS, param_count  # noqa: E402
from alphabet.char import CharTokenizer  # noqa: E402
from alphabet.bpe import BPETokenizer  # noqa: E402
from ember.format import save_ember  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIT = os.path.join(ROOT, "witness")
SAMPLES = os.path.join(ROOT, "kindling", "samples")

SCALES = {
    "seed": dict(
        kind="char", vocab="char_vocab.json",
        train_bin="witness_train.bin", val_bin="witness_val.bin",
        make=cfgs.seed, B=32, T=256, lr_max=6e-4, lr_min=6e-5, warmup=200,
        default_steps=10000, ember="logos_seed.ember", prompt="In the beginning",
    ),
    "acolyte": dict(
        kind="bpe", vocab="bpe_vocab.json",
        train_bin="witness_train_bpe.bin", val_bin="witness_val_bpe.bin",
        make=cfgs.acolyte, B=12, T=512, lr_max=3e-4, lr_min=3e-5, warmup=300,
        default_steps=60000, ember="logos_acolyte.ember", prompt="In the beginning",
    ),
}

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
BETAS = (0.9, 0.95)
EVAL_EVERY = 250
SAMPLE_EVERY = 500
CKPT_EVERY = 1000


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
            p = p - lr * (m / bc1 / (mx.sqrt(v / bc2) + self.eps) + self.wd * p)
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
        s = max_norm / (tn + 1e-6)
        grads = tree_unflatten([(n, g * s) for n, g in tree_flatten(grads)])
    return grads, tn


def make_lr(sc, max_steps):
    def lr_at(step):
        if step < sc["warmup"]:
            return sc["lr_max"] * (step + 1) / sc["warmup"]
        prog = (step - sc["warmup"]) / max(1, max_steps - sc["warmup"])
        return sc["lr_min"] + 0.5 * (sc["lr_max"] - sc["lr_min"]) * (1 + math.cos(math.pi * prog))
    return lr_at


def get_batch(data, B, T):
    ix = np.random.randint(0, len(data) - T - 1, size=B)
    x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
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


def load_tokenizer(sc):
    path = os.path.join(WIT, sc["vocab"])
    return CharTokenizer.load(path) if sc["kind"] == "char" else BPETokenizer.load(path)


def main():
    scale = "seed"
    steps = None
    args = sys.argv[1:]
    if args and args[0] in SCALES:
        scale = args[0]
        args = args[1:]
    if args:
        steps = int(args[0])
    sc = SCALES[scale]
    max_steps = steps or sc["default_steps"]
    os.makedirs(SAMPLES, exist_ok=True)

    tok = load_tokenizer(sc)
    cfg = sc["make"](tok.vocab_size)
    cfg.context = sc["T"]                       # train and serve at the same length
    train_data = np.memmap(os.path.join(WIT, sc["train_bin"]), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(WIT, sc["val_bin"]), dtype=np.uint16, mode="r")
    print("scale %s  vocab %d  ctx %d  train %d tok  val %d tok"
          % (scale, tok.vocab_size, cfg.context, len(train_data), len(val_data)))

    model = LOGOS(cfg)
    mx.eval(model.parameters())
    print("LOGOS %s: %.2f M params" % (scale, param_count(model) / 1e6))

    opt = AdamW(BETAS, WEIGHT_DECAY)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: cross_entropy(m(x), y))
    lr_at = make_lr(sc, max_steps)
    ember_out = os.path.join(ROOT, "ember", sc["ember"])

    log = open(os.path.join(SAMPLES, "kindling_%s.txt" % scale), "a", encoding="utf-8")
    t0 = time.time()
    for step in range(max_steps):
        lr = lr_at(step)
        x, y = get_batch(train_data, sc["B"], sc["T"])
        loss, grads = loss_and_grad(model, x, y)
        grads, gnorm = clip_grads(grads, GRAD_CLIP)
        opt.step(model, grads, lr)
        mx.eval(model.parameters(), loss)

        if step % EVAL_EVERY == 0 or step == max_steps - 1:
            vx, vy = get_batch(val_data, sc["B"], sc["T"])
            vl = cross_entropy(model(vx), vy)
            mx.eval(vl)
            line = "step %6d  train %.4f  val %.4f  lr %.2e  gnorm %.2f  %.1fs" % (
                step, loss.item(), vl.item(), lr, gnorm, time.time() - t0)
            print(line)
            log.write(line + "\n"); log.flush()

        if step % SAMPLE_EVERY == 0 or step == max_steps - 1:
            block = "\n----- step %d sample -----\n%s\n" % (step, sample(model, tok, cfg, sc["prompt"]))
            print(block)
            log.write(block); log.flush()

        if step > 0 and (step % CKPT_EVERY == 0 or step == max_steps - 1):
            save_ember(model, cfg, tok, ember_out)

    save_ember(model, cfg, tok, ember_out)
    log.close()
    print("done. ember at %s" % ember_out)


if __name__ == "__main__":
    main()

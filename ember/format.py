"""The Ember. The whole soul of LOGOS in one self-contained file.

Our own container, no framework save format, so the weights are portable and ours and
can be read by a from-scratch engine with no library. Layout:

  bytes 0..5   magic b'LOGOS\\0'
  byte  6      version
  uint32       config json length, then the json (model shape)
  uint32       tokenizer blob length, then the blob (the alphabet)
  per tensor:  uint16 name len, name, uint8 ndim, uint32[ndim] shape,
               uint8 dtype (0 float32), raw little-endian bytes
"""

import json
import struct

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from logos.config import Config
from logos.model import LOGOS

MAGIC = b"LOGOS\x00"
VERSION = 1
DT_F32 = 0


def _tokenizer_from_blob(blob):
    kind = json.loads(blob.decode("utf-8")).get("kind", "char")
    if kind == "char":
        from alphabet.char import CharTokenizer
        return CharTokenizer.from_blob(blob)
    raise ValueError("unknown tokenizer kind: %s" % kind)


def save_ember(model, cfg, tokenizer, path):
    params = tree_flatten(model.parameters())
    cfg_blob = json.dumps(_cfg_dict(cfg)).encode("utf-8")
    tok_blob = tokenizer.to_blob()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<B", VERSION))
        f.write(struct.pack("<I", len(cfg_blob)))
        f.write(cfg_blob)
        f.write(struct.pack("<I", len(tok_blob)))
        f.write(tok_blob)
        for name, arr in params:
            a = np.array(arr.astype(mx.float32), copy=False).astype("<f4")
            nb = name.encode("utf-8")
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<B", a.ndim))
            f.write(struct.pack("<%dI" % a.ndim, *a.shape))
            f.write(struct.pack("<B", DT_F32))
            f.write(a.tobytes())


def load_ember(path):
    with open(path, "rb") as f:
        buf = f.read()
    off = 0
    assert buf[:6] == MAGIC, "not an Ember file"
    off = 6
    (version,) = struct.unpack_from("<B", buf, off); off += 1
    assert version == VERSION, "Ember version mismatch"
    (clen,) = struct.unpack_from("<I", buf, off); off += 4
    cfg = Config(**json.loads(buf[off:off + clen].decode("utf-8"))); off += clen
    (tlen,) = struct.unpack_from("<I", buf, off); off += 4
    tokenizer = _tokenizer_from_blob(buf[off:off + tlen]); off += tlen

    tensors = []
    n = len(buf)
    while off < n:
        (nlen,) = struct.unpack_from("<H", buf, off); off += 2
        name = buf[off:off + nlen].decode("utf-8"); off += nlen
        (ndim,) = struct.unpack_from("<B", buf, off); off += 1
        shape = struct.unpack_from("<%dI" % ndim, buf, off); off += 4 * ndim
        (dt,) = struct.unpack_from("<B", buf, off); off += 1
        assert dt == DT_F32, "unsupported dtype %d" % dt
        count = 1
        for s in shape:
            count *= s
        nbytes = count * 4
        a = np.frombuffer(buf, dtype="<f4", count=count, offset=off).reshape(shape)
        off += nbytes
        tensors.append((name, mx.array(np.array(a))))

    model = LOGOS(cfg)
    model.update(tree_unflatten(tensors))
    mx.eval(model.parameters())
    return model, cfg, tokenizer


def _cfg_dict(cfg):
    return {
        "vocab": cfg.vocab,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "d_model": cfg.d_model,
        "d_ff": cfg.d_ff,
        "context": cfg.context,
        "rope_theta": cfg.rope_theta,
    }

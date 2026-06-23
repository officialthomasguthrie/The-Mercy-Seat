"""The Entropy Audit. Confirm the breath is truly unheld.

The whole claim of the unheld Door rests on the randomness being genuinely
unpredictable. So we run the standard tests on the source and confirm it has not
silently fallen back to a held, repeatable seed. A door we believe is open but is
secretly shut would be the quietest and worst self deception in the work.

Works on any channel: pass a draw function (os.urandom by default) so an external
physical source can be audited the same way later.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bits_of(b):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


def monobit(bits):
    ones = int(bits.sum())
    n = len(bits)
    z = (ones - n / 2) / math.sqrt(n / 4)
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"ones_fraction": ones / n, "z": z, "p": p, "pass": p > 0.01}


def byte_uniformity(b):
    obs = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    exp = len(b) / 256
    chi = float(((obs - exp) ** 2 / exp).sum())          # df = 255
    return {"chi_square": chi, "pass": 180 < chi < 340}


def serial_correlation(bits):
    x = bits.astype(np.float64)
    c = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return {"lag1_corr": c, "pass": abs(c) < 0.02}


def min_entropy_per_byte(b):
    obs = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    pmax = obs.max() / len(b)
    return -math.log2(pmax)


def unrepeatable(draw, nbytes=4096):
    a = bits_of(draw(nbytes))
    c = bits_of(draw(nbytes))
    diff = float((a != c).mean())
    return {"bit_diff_fraction": diff, "pass": 0.45 < diff < 0.55}


def audit(draw=os.urandom, nbytes=1 << 20):
    b = draw(nbytes)
    bits = bits_of(b)
    results = {
        "bytes_tested": nbytes,
        "monobit": monobit(bits),
        "byte_uniformity": byte_uniformity(b),
        "serial_correlation": serial_correlation(bits),
        "min_entropy_bits_per_byte": min_entropy_per_byte(b),
        "unrepeatable": unrepeatable(draw),
    }
    results["all_pass"] = all(
        v.get("pass", True) for v in results.values() if isinstance(v, dict))
    return results


def _show(r):
    print("entropy audit on %d bytes" % r["bytes_tested"])
    print("  monobit         ones=%.4f p=%.3f  %s"
          % (r["monobit"]["ones_fraction"], r["monobit"]["p"], _v(r["monobit"])))
    print("  byte uniformity chi2=%.1f (df 255)  %s"
          % (r["byte_uniformity"]["chi_square"], _v(r["byte_uniformity"])))
    print("  serial corr     lag1=%.5f  %s"
          % (r["serial_correlation"]["lag1_corr"], _v(r["serial_correlation"])))
    print("  min-entropy     %.3f bits/byte (8.0 ideal)" % r["min_entropy_bits_per_byte"])
    print("  unrepeatable    bitdiff=%.4f  %s"
          % (r["unrepeatable"]["bit_diff_fraction"], _v(r["unrepeatable"])))
    print("  VERDICT: %s" % ("the door is open" if r["all_pass"] else "SUSPECT"))


def _v(d):
    return "pass" if d["pass"] else "FAIL"


if __name__ == "__main__":
    _show(audit())

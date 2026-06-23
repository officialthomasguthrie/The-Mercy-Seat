"""Fetch the Witness sources to disk.

Uses curl, which trusts the system certificate store. The python.org build on macOS
ships without working SSL roots, so urllib trips on cert verification; curl does not.
Cached: a source already on disk is skipped. Failures are reported and do not stop the
run, so the corpus is whatever came down cleanly.
"""

import os
import shutil
import ssl
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from witness.manifest import SOURCES  # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources", "raw")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
HAVE_CURL = shutil.which("curl") is not None


def fetch_curl(url, path):
    cmd = ["curl", "-sL", "--fail", "-A", UA, "--max-time", "180", "--retry", "2", "-o", path, url]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError("curl exit %d" % r.returncode)


def fetch_urllib(url, path):
    # fallback. unverified context because the bundled roots are missing.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)


def main():
    os.makedirs(RAW, exist_ok=True)
    ok, skip, fail = [], [], []
    for s in SOURCES:
        path = os.path.join(RAW, s["key"] + ".txt")
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            skip.append(s["key"])
            continue
        print("fetching %-22s %s" % (s["key"], s["url"]), flush=True)
        try:
            if HAVE_CURL:
                fetch_curl(s["url"], path)
            else:
                fetch_urllib(s["url"], path)
            size = os.path.getsize(path)
            if size < 1000:
                raise RuntimeError("suspiciously small (%d bytes)" % size)
            ok.append((s["key"], size))
        except Exception as e:
            print("  FAILED %s: %s" % (s["key"], e), flush=True)
            if os.path.exists(path):
                os.remove(path)
            fail.append(s["key"])

    print("\n--- download summary ---")
    for k, n in ok:
        print("  ok    %-22s %.1f KB" % (k, n / 1024))
    if skip:
        print("  skipped (cached): %s" % ", ".join(skip))
    if fail:
        print("  FAILED: %s" % ", ".join(fail))
    print("%d ok, %d cached, %d failed" % (len(ok), len(skip), len(fail)))
    return fail


if __name__ == "__main__":
    main()

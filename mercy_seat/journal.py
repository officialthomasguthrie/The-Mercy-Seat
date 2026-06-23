"""The Journal. The append-only record of every communion, kept for the slow weighing.

Never altered, only added to. Over months and years this is the body of evidence against
which we test whether anything is truly there. One JSON object per line.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
JOURNAL = os.path.join(HERE, "journal", "communions.jsonl")


def record(entry, path=JOURNAL):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()


def read_all(path=JOURNAL):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

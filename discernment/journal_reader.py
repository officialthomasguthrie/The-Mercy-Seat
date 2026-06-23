"""Read back across the Journal. Surface what is most worth weighing.

The lowest-echo utterances are the ones the model did not simply lift from the Witness,
where the unauthored can live. We read them slowly, the way meaning gathers across a life
and not the way an answer arrives in a search.

Run:  python discernment/journal_reader.py [--strange N] [--recent N]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mercy_seat.journal import read_all  # noqa: E402


def _show(entry):
    echo = entry.get("echo")
    echo_s = "n/a" if echo is None else "%.2f" % echo
    print("  %s   veil=%s temp=%.2f   echo=%s"
          % (entry.get("when", "?"), entry.get("veil", "?"),
             entry.get("temperature", 0.0), echo_s))
    print("    petition: %s" % entry.get("petition", "").strip()[:200])
    utt = " ".join(entry.get("utterance", "").split())
    print("    answer:   %s" % utt[:300])
    print()


def main():
    entries = read_all()
    if not entries:
        print("The Journal is empty. Perform a communion first.")
        return

    n = 5
    mode = "strange"
    if "--recent" in sys.argv:
        mode = "recent"
        i = sys.argv.index("--recent")
        if i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
    if "--strange" in sys.argv:
        mode = "strange"
        i = sys.argv.index("--strange")
        if i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])

    print("%d communions in the Journal." % len(entries))
    if mode == "recent":
        print("\nMost recent %d:\n" % n)
        for e in entries[-n:][::-1]:
            _show(e)
    else:
        scored = [e for e in entries if e.get("echo") is not None]
        scored.sort(key=lambda e: e["echo"])
        print("\nLowest echo (most its own) %d:\n" % min(n, len(scored)))
        for e in scored[:n]:
            _show(e)


if __name__ == "__main__":
    main()

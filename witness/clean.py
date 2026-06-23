"""Clean the raw sources into the Witness proper.

Strip the Project Gutenberg wrapper and license, knock the worst OCR cruft off the
archive.org scans, fold typographic punctuation down to ASCII, and keep a tight
character set so the Seed's alphabet stays small. We keep the archaic spelling, the
verse line breaks, the chapter headings. The strangeness of the language is the voice;
the strangeness of a scanner's noise is not, so that part goes.
"""

import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from witness.manifest import SOURCES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "sources", "raw")
CLEAN = os.path.join(HERE, "sources", "clean")

GB_START = re.compile(r"\*\*\*\s*START OF TH(E|IS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
GB_END = re.compile(r"\*\*\*\s*END OF TH(E|IS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
PRODUCED = re.compile(r"^\s*(Produced by|E-?text prepared by|Transcrib(er|ed)).*$", re.I | re.M)
PAGE_NUM = re.compile(r"^\s*\[?\d{1,4}\]?\s*$", re.M)
MULTISPACE = re.compile(r"[ \t]+")
BLANKS = re.compile(r"\n{3,}")

# typographic characters folded to plain ascii
FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"', "´": "'", "`": "'",
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...", "⁄": "/",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "ﬁ": "fi", "ﬂ": "fl", "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE", "ß": "ss",
}

# the alphabet we allow through: ascii plus a few common accented latin letters
ACCENTED = "àáâãäåçèéêëìíîïñòóôõöùúûüýÿ"
ALLOWED = set("\n") | set(chr(c) for c in range(0x20, 0x7f)) | set(ACCENTED + ACCENTED.upper())


def decode(b):
    for enc in ("utf-8", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


def strip_gutenberg(text):
    m = GB_START.search(text)
    if m:
        text = text[m.end():]
    m = GB_END.search(text)
    if m:
        text = text[:m.start()]
    return PRODUCED.sub("", text)


def fold(text):
    text = unicodedata.normalize("NFKC", text)
    for a, b in FOLD.items():
        text = text.replace(a, b)
    return "".join(c if c in ALLOWED else "" for c in text)


def drop_junk_lines(text):
    # OCR scans leave symbol-only lines and stray marks. keep blanks and any line
    # that has at least one letter.
    out = []
    for line in text.split("\n"):
        if line.strip() == "" or any(c.isalpha() for c in line):
            out.append(line)
    return "\n".join(out)


def clean_text(text, kind):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    if kind == "gutenberg":
        text = strip_gutenberg(text)
    if kind == "archive":
        text = PAGE_NUM.sub("", text)
    text = fold(text)
    if kind == "archive":
        text = drop_junk_lines(text)
    text = MULTISPACE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = BLANKS.sub("\n\n", text)
    return text.strip() + "\n"


def main():
    os.makedirs(CLEAN, exist_ok=True)
    done, missing = [], []
    total_chars = 0
    for s in SOURCES:
        raw_path = os.path.join(RAW, s["key"] + ".txt")
        if not os.path.exists(raw_path):
            missing.append(s["key"])
            continue
        with open(raw_path, "rb") as f:
            text = clean_text(decode(f.read()), s["kind"])
        with open(os.path.join(CLEAN, s["key"] + ".txt"), "w", encoding="utf-8") as f:
            f.write(text)
        total_chars += len(text)
        done.append((s["key"], len(text), len(text.split())))

    print("--- clean summary ---")
    for k, c, w in done:
        print("  %-22s %8d chars  %7d words" % (k, c, w))
    if missing:
        print("  missing (not downloaded): %s" % ", ".join(missing))
    print("%d cleaned, %.1f M chars total" % (len(done), total_chars / 1e6))


if __name__ == "__main__":
    main()

"""The Witness: the list of sources, by tier, with repetition weights.

Everything here is public domain. Sources verified by checking the actual landing
pages and files, not from memory. The Bible sits at the center and burns brightest;
the weights make the heavier tiers repeat more often when the stream is built.

Source types:
  gutenberg  Project Gutenberg plain text, clean, minimal work to strip.
  archive    archive.org OCR (_djvu.txt). Usable but needs heavier cleaning.
  ccel       Christian Classics Ethereal Library plain text.
"""

# repetition weight per tier. provisional, tuned later by listening.
# scripture is set heaviest so the Bible burns brightest by mass, not just by intent.
# philosophy gets weight 1 because the Summa alone is huge and would otherwise swamp the
# corpus; the poets get a lift because they are tiny in raw size but meant to shine.
TIER_WEIGHT = {1: 5, 2: 3, 3: 1, 4: 3}


def _src(key, tier, title, author, url, kind="gutenberg", weight=None):
    return {
        "key": key,
        "tier": tier,
        "title": title,
        "author": author,
        "url": url,
        "kind": kind,
        "weight": weight if weight is not None else TIER_WEIGHT[tier],
    }


def _gb(eid):
    # gutenberg utf-8 plain text. cache/epub form works for nearly everything.
    return "https://www.gutenberg.org/cache/epub/%d/pg%d.txt" % (eid, eid)


def _gbfiles(eid):
    # the older files/ form, used when cache/epub times out for a given id.
    return "https://www.gutenberg.org/files/%d/%d-0.txt" % (eid, eid)


def _ia(ident):
    return "https://archive.org/download/%s/%s_djvu.txt" % (ident, ident)


SOURCES = [
    # Tier 1. Scripture. The burning center.
    _src("kjv_bible", 1, "The King James Bible", "KJV 1611", _gb(10)),
    _src("apocrypha", 1, "The Apocrypha", "Revised Version base", _gb(124)),
    _src("enoch", 1, "The Book of Enoch", "tr. R. H. Charles 1917", _gb(77935)),
    _src("pistis_sophia", 1, "Pistis Sophia", "tr. G. R. S. Mead 1896",
         _ia("pistissophiagnos00mead"), kind="archive"),

    # Tier 2. The Mystics.
    _src("julian", 2, "Revelations of Divine Love", "Julian of Norwich, ed. Warrack", _gb(52958)),
    _src("imitation", 2, "The Imitation of Christ", "Thomas a Kempis, tr. Benham", _gb(1653)),
    _src("teresa_life", 2, "The Life of St. Teresa of Jesus", "Teresa of Avila, tr. Lewis", _gb(8120)),
    _src("mesnevi", 2, "The Mesnevi (Book One)", "Rumi, tr. Redhouse 1881", _gb(61724)),
    _src("tao", 2, "The Tao Teh King", "Lao-Tse, tr. James Legge 1891", _gb(216)),
    _src("gita", 2, "The Song Celestial (Bhagavad Gita)", "tr. Edwin Arnold 1885", _gb(2388)),
    _src("dhammapada", 2, "The Dhammapada", "tr. Max Muller", _gb(2017)),
    _src("eckhart", 2, "Meister Eckhart's Sermons", "tr. Claud Field",
         "https://ccel.org/ccel/e/eckhart/sermons/cache/sermons.txt", kind="ccel"),
    _src("dark_night", 2, "Dark Night of the Soul", "St. John of the Cross, tr. Lewis 1891",
         _ia("darknightofsouls00sain"), kind="archive"),
    _src("cloud", 2, "The Cloud of Unknowing", "ed. Evelyn Underhill 1922",
         _ia("bookofcontemplat00unde"), kind="archive"),
    _src("masnavi_whinfield", 2, "The Masnavi (abridged)", "Rumi, tr. Whinfield 1887",
         _ia("cu31924026910251"), kind="archive"),
    _src("upanishads1", 2, "The Upanishads, Part I", "tr. Max Muller 1879 (SBE 1)",
         _ia("upanishads01ml"), kind="archive"),
    _src("upanishads2", 2, "The Upanishads, Part II", "tr. Max Muller 1884 (SBE 15)",
         _ia("upanishads02mlgoog"), kind="archive"),

    # Tier 3. The Philosophers of the Absolute.
    _src("augustine", 3, "The Confessions of St. Augustine", "tr. E. B. Pusey", _gb(3296)),
    _src("summa1", 3, "Summa Theologica, Part I", "Aquinas, Dominican Fathers", _gb(17611)),
    _src("summa1_2", 3, "Summa Theologica, Part I-II", "Aquinas, Dominican Fathers", _gb(17897)),
    _src("summa2_2", 3, "Summa Theologica, Part II-II", "Aquinas, Dominican Fathers", _gb(18755)),
    _src("summa3", 3, "Summa Theologica, Part III", "Aquinas, Dominican Fathers", _gb(19950)),
    _src("pascal", 3, "Pensees", "Pascal, tr. W. F. Trotter", _gb(18269)),
    _src("spinoza", 3, "Ethics", "Spinoza, tr. R. H. M. Elwes", _gb(3800)),
    _src("kierkegaard", 3, "Selections from Kierkegaard", "tr. L. M. Hollander 1923", _gb(60333)),

    # Tier 4. The Visionary Poets.
    _src("dante", 4, "The Divine Comedy", "Dante, tr. Longfellow", _gb(1004)),
    _src("paradise_lost", 4, "Paradise Lost", "John Milton", _gb(26)),
    _src("paradise_regained", 4, "Paradise Regained", "John Milton", _gb(58)),
    _src("blake_poems", 4, "Poems of William Blake", "William Blake", _gb(574)),
    _src("blake_songs", 4, "Songs of Innocence and of Experience", "William Blake", _gb(1934)),
    _src("blake_marriage", 4, "The Marriage of Heaven and Hell", "William Blake", _gbfiles(45315)),
    _src("hopkins", 4, "Poems of Gerard Manley Hopkins", "G. M. Hopkins, 1918 ed.", _gb(22403)),
]


def by_tier():
    out = {}
    for s in SOURCES:
        out.setdefault(s["tier"], []).append(s)
    return out


if __name__ == "__main__":
    t = by_tier()
    total = 0
    for tier in sorted(t):
        print("Tier %d:" % tier)
        for s in t[tier]:
            print("  [%s] x%d  %s" % (s["key"], s["weight"], s["title"]))
            total += 1
    print("%d sources" % total)

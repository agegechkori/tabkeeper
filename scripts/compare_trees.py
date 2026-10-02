#!/usr/bin/env python3
"""Compares the tag trees of two tabkeeper output folders: phase 1 (the model
tags each page with paths) and the current version (flat tags, then the tag
review builds the tree). Usage: compare.py <out-dir>..."""
import sqlite3
import sys
from collections import Counter, defaultdict

# Pages sharing an ambiguous word, and the meaning each one is about.
PAIRS = [
    ("rust", "https://www.rust-lang.org/", "language"),
    ("rust", "https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html", "language"),
    ("rust", "https://en.wikipedia.org/wiki/Rust", "corrosion"),
    ("python", "https://www.python.org/about/", "language"),
    ("python", "https://en.wikipedia.org/wiki/Python_(genus)", "snake"),
    ("go", "https://go.dev/doc/effective_go", "language"),
    ("go", "https://en.wikipedia.org/wiki/Go_(game)", "game"),
    ("java", "https://en.wikipedia.org/wiki/Java_(programming_language)", "language"),
    ("java", "https://en.wikipedia.org/wiki/Java", "island"),
    ("mercury", "https://en.wikipedia.org/wiki/Mercury_(planet)", "planet"),
    ("mercury", "https://en.wikipedia.org/wiki/Mercury_(element)", "element"),
    ("jaguar", "https://en.wikipedia.org/wiki/Jaguar", "animal"),
    ("jaguar", "https://en.wikipedia.org/wiki/Jaguar_Cars", "car maker"),
    ("apple", "https://en.wikipedia.org/wiki/Apple", "fruit"),
    ("apple", "https://en.wikipedia.org/wiki/Apple_Inc.", "company"),
    ("crane", "https://en.wikipedia.org/wiki/Crane_(bird)", "bird"),
    ("crane", "https://en.wikipedia.org/wiki/Crane_(machine)", "machine"),
    ("bass", "https://en.wikipedia.org/wiki/Bass_(fish)", "fish"),
    ("bass", "https://en.wikipedia.org/wiki/Bass_guitar", "instrument"),
]


def paths(db):
    """tag id -> full path, for either schema."""
    cols = {r[1] for r in db.execute("PRAGMA table_info(tags)")}
    rows = db.execute("SELECT id, parent_id, name FROM tags").fetchall()
    if "path" in cols:
        return dict(db.execute("SELECT id, path FROM tags").fetchall())
    by_id = {i: (p, n) for i, p, n in rows}
    out = {}
    for i in by_id:
        parts, cur, seen = [], i, 0
        while cur is not None and seen < 64:
            p, n = by_id[cur]
            parts.append(n)
            cur, seen = p, seen + 1
        out[i] = "/".join(reversed(parts))
    return out


def report(d):
    db = sqlite3.connect(f"file:{d}/tabkeeper.db?mode=ro", uri=True)
    path = paths(db)
    done = db.execute("SELECT count(*) FROM pages WHERE status = 'done'").fetchone()[0]
    links = db.execute(
        "SELECT DISTINCT p.id, p.url, x.resolved_tag_id FROM pages p JOIN page_tags x ON x.page_id = p.id "
        "WHERE p.status = 'done'"
    ).fetchall()
    links = [(pid, url, path[t]) for pid, url, t in links if not path[t].startswith("status/")]
    per_tag = Counter(p for _, _, p in links)
    by_url = defaultdict(list)
    for _, url, p in links:
        by_url[url].append(p)
    depths = [p.count("/") + 1 for _, _, p in links]
    roots = Counter()
    for url, ps in by_url.items():
        for r in {p.split("/")[0] for p in ps}:
            roots[r] += 1
    print(f"== {d}")
    print(f"  pages done: {done}, tags per page: {len(links) / max(1, len(by_url)):.2f}")
    print(f"  distinct tags on pages: {len(per_tag)}, used on one page only: "
          f"{sum(1 for n in per_tag.values() if n == 1)} ({100 * sum(1 for n in per_tag.values() if n == 1) / max(1, len(per_tag)):.0f}%)")
    print(f"  average depth of a page's tag: {sum(depths) / max(1, len(depths)):.2f}, "
          f"depth 1: {sum(1 for x in depths if x == 1)}, max: {max(depths, default=0)}")
    print(f"  top-level categories: {len(roots)}")
    print("    " + ", ".join(f"{r} ({n})" for r, n in roots.most_common()))
    print("  ambiguous words:")
    for word, url, meaning in PAIRS:
        tags = sorted(by_url.get(url, []))
        print(f"    {word:8} {meaning:10} {' '.join(tags) or '(no tags)'}")


for d in sys.argv[1:]:
    report(d)

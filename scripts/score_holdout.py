"""Score an archive against scripts/holdout_urls.txt.

Topics of 3+ pages: cluster share, pairwise precision/recall of the first tag,
first tags from another topic, collisions on ambiguous words.
Pairs and one-off pages: listed with their tags, for judging by hand."""
import collections, itertools, os, sqlite3, sys

LABELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holdout_urls.txt")


def read_labels():
    clusters, ambiguous, cur = {}, [], None
    for line in open(LABELS):
        line = line.strip()
        if line.startswith("# cluster:"):
            cur = line.split(":", 1)[1].strip()
        elif line.startswith("# ambiguous:"):
            ambiguous.append(tuple(line.split(":", 1)[1].split()))
        elif line.startswith("http") and cur:
            clusters.setdefault(cur, []).append(line.split()[0])
    return clusters, ambiguous


def tags_of(db_path):
    db = sqlite3.connect(db_path)
    cols = [r[1] for r in db.execute("pragma table_info(page_tags)")]
    order = "pt.position" if "position" in cols else "pt.rowid"
    rows = db.execute(
        "select p.url, t.name from page_tags pt join pages p on p.id = pt.page_id "
        f"join tags t on t.id = pt.resolved_tag_id order by p.id, {order}"
    ).fetchall()
    out = collections.defaultdict(list)
    for url, tag in rows:
        if tag not in out[url]:
            out[url].append(tag)
    return out


def score(d, verbose):
    clusters, ambiguous = read_labels()
    tags = tags_of(f"{d}/tabkeeper.db")
    topic_of = {u: c for c, us in clusters.items() for u in us}
    topics = {c: us for c, us in clusters.items() if len(us) >= 3}
    pages = [u for us in topics.values() for u in us]
    first = {u: tags[u][0] if tags[u] else None for u in pages}

    shares, rows = [], []
    for c, us in topics.items():
        counts = collections.Counter(t for u in us for t in tags[u])
        best, n = counts.most_common(1)[0] if counts else ("-", 0)
        shares.append(n / len(us))
        rows.append(f"    {c:18} {n}/{len(us)} {best:20} others: {', '.join(t for t, _ in counts.most_common()[1:5])}")

    majority = {}
    by_tag = collections.defaultdict(collections.Counter)
    for u in pages:
        for t in tags[u]:
            by_tag[t][topic_of[u]] += 1
    for t, c in by_tag.items():
        majority[t] = c.most_common(1)[0][0]

    same = sum(1 for a, b in itertools.combinations(pages, 2) if topic_of[a] == topic_of[b])
    shared = [(a, b) for a, b in itertools.combinations(pages, 2) if first[a] and first[a] == first[b]]
    tp = sum(1 for a, b in shared if topic_of[a] == topic_of[b])
    p = tp / len(shared) if shared else 0
    r = tp / same
    off = [(u.rsplit("/", 1)[-1][:28], topic_of[u], first[u]) for u in pages if first[u] and majority[first[u]] != topic_of[u]]
    extra = [t for u in pages for t in tags[u][1:] if majority[t] != topic_of[u]]
    collisions = []
    for a, b in ambiguous:
        ta = {t for u in clusters[a] for t in tags[u]}
        tb = {t for u in clusters[b] for t in tags[u]}
        collisions += [f"{t} ({a} + {b})" for t in sorted(ta & tb)]

    all_urls = [u for us in clusters.values() for u in us]
    untagged = [u for u in all_urls if not tags[u]]
    distinct = {t for u in all_urls for t in tags[u]}
    per_page = sum(len(tags[u]) for u in all_urls) / max(1, len(all_urls) - len(untagged))
    print(f"== {d}")
    print(f"  topics (3+ pages): cluster share {sum(shares) / len(shares):.2f}; first tag pairwise "
          f"precision {p:.2f} recall {r:.2f} F1 {2 * p * r / (p + r) if p + r else 0:.2f}")
    print(f"  first tag from another topic: {len(off)}; extra tags from another topic: {len(extra)} "
          f"{collections.Counter(extra).most_common(6)}")
    print(f"  ambiguous-word collisions: {len(collisions)} {collisions}")
    print(f"  all 150: untagged {len(untagged)}, distinct tags {len(distinct)}, tags per tagged page {per_page:.2f}")
    if verbose:
        print("  per topic:")
        print("\n".join(rows))
        for o in off:
            print("    off-topic first tag:", o)
        print("  pairs and one-off pages:")
        for c, us in clusters.items():
            if len(us) < 3:
                for u in us:
                    print(f"    {c:24} {u.rsplit('/', 1)[-1][:30]:30} {', '.join(tags[u]) or '(none)'}")


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    for d in [a for a in sys.argv[1:] if a != "-v"]:
        score(d, verbose)

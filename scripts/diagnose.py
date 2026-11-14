"""Where a review's tagging goes wrong, and how much reviews agree.

    python3 scripts/diagnose.py --labels holdout|old|FILE ARCHIVE... [--stability] [-v]

For each archive, on the topics of 3+ pages: first-tag pairwise F1, first
tags from another topic, untagged pages by cause, and topics lost
altogether (no page kept a tag of the topic's own). If the review's trace
is next to the archive (ARCHIVE.jsonl, from TABKEEPER_TRACE), it also says
whether the topic's tag was offered to the classifier, and whether the
model answered none although it was. A topic's tag is the tag most of its
pages got first, among tags mostly used on that topic.

--stability compares every pair of archives: how often two pages that
share a first tag in one also share one in the other (pairwise F1 of the
two first-tag groupings, over all pages)."""
import argparse
import collections
import itertools
import json
import os
import re
import sqlite3
import statistics

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
LABELS = {"holdout": "holdout_urls.txt", "old": "cluster_urls.txt"}


def read_labels(path):
    clusters, cur = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith("# cluster:"):
            cur = line.split(":", 1)[1].strip()
        elif line.startswith("http") and cur:
            clusters.setdefault(cur, []).append(line.split()[0])
    return clusters


def read_archive(path):
    db = sqlite3.connect(f"file:{path}/tabkeeper.db?mode=ro", uri=True)
    cols = [r[1] for r in db.execute("pragma table_info(page_tags)")]
    order = "pt.position" if "position" in cols else "pt.rowid"
    tags = collections.defaultdict(list)
    for url, tag in db.execute(
        "select p.url, t.name from page_tags pt join pages p on p.id = pt.page_id "
        "join tags t on t.id = pt.resolved_tag_id where t.name not like 'status/%' "
        f"order by p.id, {order}"
    ):
        if tag not in tags[url]:
            tags[url].append(tag)
    pages = db.execute("select url, coalesce(title, ''), coalesce(summary, '') from pages").fetchall()
    return tags, pages


def read_requests(trace, pages):
    """The classification requests of each page, in order: (offered, picked)."""
    key = {(title, summary[:120]): url for url, title, summary in pages}
    requests = collections.defaultdict(list)
    for line in open(trace):
        record = json.loads(line)
        if record["name"] != "page_tags":
            continue
        m = re.match(r"Page: (.*)\n(.*?)\n\nTags to choose from:\n(.*)", record["user"], re.S)
        if not m:
            continue
        offered = [l[2:].split(":")[0] for l in m.group(3).splitlines() if l.startswith("- ")]
        url = key.get((m.group(1), m.group(2)[:120]))
        if url:
            requests[url].append((offered, (record.get("reply") or {}).get("tags") or []))
    return requests


def first_tag_pairs(urls, first):
    return {(a, b) for a, b in itertools.combinations(sorted(urls), 2) if first.get(a) and first.get(a) == first.get(b)}


def diagnose(path, clusters):
    tags, pages = read_archive(path)
    topic_of = {u: c for c, us in clusters.items() for u in us}
    scored = {c: us for c, us in clusters.items() if len(us) >= 3}
    scored_pages = [u for us in scored.values() for u in us]
    first = {u: (tags[u][0] if tags[u] else None) for u in topic_of}

    by_tag = collections.defaultdict(collections.Counter)
    for u in scored_pages:
        for t in tags[u]:
            by_tag[t][topic_of[u]] += 1
    majority = {t: c.most_common(1)[0][0] for t, c in by_tag.items()}
    home = {}
    for c, us in scored.items():
        firsts = collections.Counter(first[u] for u in us if first[u] and majority[first[u]] == c)
        home[c] = firsts.most_common(1)[0][0] if firsts else None

    same = sum(1 for a, b in itertools.combinations(scored_pages, 2) if topic_of[a] == topic_of[b])
    shared = first_tag_pairs(scored_pages, first)
    tp = sum(1 for a, b in shared if topic_of[a] == topic_of[b])
    precision = tp / len(shared) if shared else 0
    recall = tp / same if same else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    trace = path.rstrip("/") + ".jsonl"
    requests = read_requests(trace, pages) if os.path.exists(trace) else None
    out = collections.Counter()
    notes = []
    offered = ranked_first = with_home = 0
    for u in scored_pages:
        c, f, h = topic_of[u], first[u], home[topic_of[u]]
        rq = requests.get(u, []) if requests is not None else None
        if h and rq:
            with_home += 1
            offered += h in rq[0][0]
            ranked_first += bool(rq[0][0]) and rq[0][0][0] == h
        if f and majority[f] == c:
            out["right"] += 1
        elif f:
            out["another topic's tag first"] += 1
            notes.append(f"another topic's tag first: {u.rsplit('/', 1)[-1][:30]} ({c}) got {f}")
        elif h is None:
            out["untagged: topic lost"] += 1
        elif rq is None:
            out["untagged: other"] += 1
        elif not rq or h not in rq[0][0]:
            out["untagged: its tag not offered"] += 1
        elif rq[0][1] in ([], ["none"]):
            out["untagged: none, its tag offered"] += 1
            notes.append(f"none although offered: {u.rsplit('/', 1)[-1][:30]} ({c}), {h} at rank {rq[0][0].index(h) + 1}")
        else:
            out["untagged: other"] += 1
    lost = sorted(c for c in scored if home[c] is None)
    all_pages = list(topic_of)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "outcomes": out,
        "lost": lost,
        "coverage": sum(1 for u in all_pages if tags[u]) / len(all_pages),
        "retrieval": (offered, ranked_first, with_home) if requests is not None else None,
        "first": first,
        "notes": notes,
        "scored": len(scored_pages),
    }


def stability(a, b, urls):
    """Pairwise F1 of the first-tag groupings of two reviews."""
    pa, pb = first_tag_pairs(urls, a), first_tag_pairs(urls, b)
    if not pa or not pb:
        return 0.0
    both = len(pa & pb)
    p, r = both / len(pb), both / len(pa)
    return 2 * p * r / (p + r) if p + r else 0.0


def spread(values, fmt):
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(statistics.mean(values))} ({fmt.format(min(values))}–{fmt.format(max(values))})"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labels", required=True, help="holdout, old, or a labels file")
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--stability", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    path = os.path.join(SCRIPTS, LABELS[args.labels]) if args.labels in LABELS else args.labels
    clusters = read_labels(path)

    results = []
    for archive in args.archives:
        r = diagnose(archive, clusters)
        results.append(r)
        o = r["outcomes"]
        untagged = sum(n for k, n in o.items() if k.startswith("untagged"))
        other = o["another topic's tag first"]
        line = (f"{archive}: F1 {r['f1']:.2f}, another topic first {other}, "
                f"untagged {untagged} of {r['scored']}, lost topics {len(r['lost'])}, coverage {r['coverage']:.0%}")
        if r["retrieval"]:
            offered, ranked_first, with_home = r["retrieval"]
            line += f", tag offered {offered}/{with_home}"
        print(line)
        if args.verbose:
            for k, n in sorted(o.items()):
                print(f"    {k}: {n}")
            if r["lost"]:
                print(f"    lost: {', '.join(r['lost'])}")
            for n in r["notes"]:
                print(f"    {n}")

    if len(results) > 1:
        print(f"== {len(results)} archives")
        print(f"  F1 {spread([r['f1'] for r in results], '{:.2f}')}")
        keys = sorted({k for r in results for k in r["outcomes"]})
        for k in keys:
            print(f"  {k}: {spread([r['outcomes'][k] for r in results], '{:.1f}')}")
        print(f"  lost topics: {spread([len(r['lost']) for r in results], '{:.1f}')}")
        print(f"  coverage: {spread([r['coverage'] for r in results], '{:.0%}')}")
        none_offered = sum(r["outcomes"]["untagged: none, its tag offered"] for r in results)
        print(f"  none although its tag was offered, in all: {none_offered}")
    if args.stability and len(results) > 1:
        urls = list(results[0]["first"])
        values = [stability(a["first"], b["first"], urls) for a, b in itertools.combinations(results, 2)]
        print(f"  stability (pairwise F1 between reviews): {spread(values, '{:.2f}')}")


if __name__ == "__main__":
    main()

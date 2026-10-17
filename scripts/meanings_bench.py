"""Benchmark designs for the "does this word have several meanings?" check
on a labeled set: real tags from the archive plus synthetic groups made
from the test clusters' page titles.
Usage: COND=a|b|c|d|e REPS=n python3 meanings_bench.py"""
import json, os, sqlite3, time, urllib.request
from collections import defaultdict

ROOT = "/Users/agegechkori/prj/tabkeeper/target/experiments/5a"
clusters, order, cur = defaultdict(list), [], None
for line in open(f"{ROOT}/urls.txt"):
    line = line.strip()
    if line.startswith("# cluster:"):
        cur = line.split(":", 1)[1].strip()
    elif line and not line.startswith("#"):
        clusters[cur].append(line)
db = sqlite3.connect(f"file:{ROOT}/p/tabkeeper.db?mode=ro", uri=True)
title_of = dict(db.execute("SELECT original_url, title FROM pages"))
summary_of = dict(db.execute("SELECT title, summary FROM pages"))


def titles(cluster, n=4, skip=()):
    return [title_of[u] for u in clusters[cluster] if u in title_of and u not in skip][:n]


W = "https://en.wikipedia.org/wiki/"
# word -> (has several meanings?, groups of titles)
CASES = {
    "python": (True, [titles("python-language"), [title_of[W + "Python_(genus)"]]]),
    "java": (True, [titles("java-language"), [title_of[W + "Java"]]]),
    "go": (True, [titles("go-language"), [title_of[W + "Go_(game)"]]]),
    "mercury": (True, [[title_of[W + "Mercury_(planet)"]], [title_of[W + "Mercury_(element)"]]]),
    "apple": (True, [titles("apple-company"), [title_of[W + "Apple"]]]),
    "jaguar": (True, [[title_of[W + "Jaguar"]], [title_of[W + "Jaguar_Cars"]]]),
    "crane": (True, [[title_of[W + "Crane_(bird)"]], [title_of[W + "Crane_(machine)"]]]),
    "bass": (True, [[title_of[W + "Bass_(fish)"]], [title_of[W + "Bass_guitar"]]]),
    "rust": (True, [titles("rust-language", 3), [title_of[W + "Rust"]]]),
    "japan": (True, [titles("japan"), titles("indonesia")]),
    "food": (False, [titles("bread"), titles("fruit"), titles("coffee", 3)]),
    "technology": (False, [titles("apple-company", 3), titles("cars", 3)]),
    "music": (False, [[title_of[W + "Jazz"], title_of[W + "Blues"]], [title_of[W + "Violin"], title_of[W + "Piano"]], [title_of[W + "Bass_guitar"]]]),
    "science": (False, [titles("astronomy"), titles("chemistry")]),
    "history": (False, [[title_of[W + "French_Revolution"], title_of[W + "Napoleon"]], [title_of[W + "Roman_Empire"]], [title_of[W + "Industrial_Revolution"]]]),
    "animals": (False, [titles("big-cats"), titles("snakes"), titles("birds")]),
    "investing": (False, [titles("investing"), [title_of[W + "Tesla,_Inc."]]]),
    "programming": (False, [titles("rust-language", 3), titles("python-language", 3), titles("java-language", 3)]),
    "plant": (False, [titles("houseplants"), [title_of[W + "Coffea_arabica"]], [title_of[W + "Pear"]]]),
    "sports": (False, [titles("running"), titles("board-games")]),
}


def ask(model, system, user, schema, think=False, max_tokens=None):
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": schema, "strict": True}},
            "temperature": 0}
    if think:
        body["max_tokens"] = 4000
    else:
        body["reasoning_effort"] = "none"
    if max_tokens:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request("http://localhost:11434/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    try:
        return json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
    except Exception:
        return None


def groups_text(word, groups):
    out = f"Tag: {word}\n"
    for g, ts in enumerate(groups):
        out += f"\nGroup {g + 1}:\n" + "".join(f"- {t}\n" for t in ts)
    return out


# (a) / (e): the engine's current prompt.
ENGINE = open(f"{ROOT}/ambiguity_prompt.txt").read()
ENGINE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["same_meaning", "groups"],
                 "properties": {"same_meaning": {"type": "boolean"}, "groups": {"type": "array", "items": {
                     "type": "object", "additionalProperties": False, "required": ["group", "tag", "description"],
                     "properties": {"group": {"type": "integer"}, "tag": {"type": "string"}, "description": {"type": "string"}}}}}}


def engine(model, word, groups, think):
    r = ask(model, ENGINE, groups_text(word, groups), ENGINE_SCHEMA, think=think)
    return None if r is None else not r["same_meaning"]


# (b) / (c): staged. Stage 1 per group: what the word refers to.
REFERENT = """In one short phrase, say what the word "{word}" refers to in these web pages. Name the thing the word itself stands for, not the topic area of the pages. Examples: "the programming language", "the snake", "the island", "the comedy troupe", "food in general".

Reply with a JSON object: {{"referent": "..."}}."""
REFERENT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["referent"], "properties": {"referent": {"type": "string"}}}
COMPARE = """A tag of a personal archive of web pages, "{word}", was found on several groups of pages. For each group you get a short phrase saying what the word refers to there.

Merge groups that refer to the same thing. Groups are the same thing when one is a kind, part or aspect of the other, or both are kinds of what the word names: phones, cars and software are all technology; bread, fruit and coffee are all food; jazz and the violin are both music. Keep groups separate only when nothing connects them except the word: python the programming language and python the snake; java the programming language and java the island; mercury the planet and mercury the chemical element.

Reply with a JSON object listing the distinct things, then which thing each group is: {{"things": [{{"id": "a", "tag": "lowercase-kebab-case-name", "description": "..."}}], "assignments": [{{"group": 1, "thing": "a"}}]}}."""
COMPARE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["things", "assignments"], "properties": {
    "things": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "tag", "description"],
               "properties": {"id": {"type": "string"}, "tag": {"type": "string"}, "description": {"type": "string"}}}},
    "assignments": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["group", "thing"],
                    "properties": {"group": {"type": "integer"}, "thing": {"type": "string"}}}}}}


def staged(model, word, groups, think):
    referents = []
    for ts in groups:
        user = f"Tag: {word}\nPage titles:\n" + "".join(f"- {t}\n" for t in ts)
        r = ask(model, REFERENT.format(word=word), user, REFERENT_SCHEMA, max_tokens=60)
        referents.append((r or {}).get("referent", "?"))
    user = "".join(f"{i + 1}. {ref}\n" for i, ref in enumerate(referents))
    r = ask(model, COMPARE.format(word=word), user, COMPARE_SCHEMA)
    if r is None:
        return None
    used = {a["thing"] for a in r["assignments"]}
    return len(used) > 1, referents


# (b2): staged, with a line of summary per page and "name the general thing".
REFERENT2 = """In one short phrase, say what the word "{word}" refers to in these web pages. Name the thing the word itself stands for, not the topic of each page. If the pages are about different examples or parts of one general thing the word names, name that general thing (for "history": "history in general"; for "music": "music in general"). Examples: "the programming language", "the snake", "the island", "the car maker", "food in general".

Reply with a JSON object: {{"referent": "..."}}."""


def first_sentence(text, n=160):
    text = (text or "").strip()
    cut = text.find(". ")
    return (text[: cut + 1] if 0 < cut < n else text[:n]).strip()


def staged2(model, word, groups, think):
    referents = []
    for ts in groups:
        user = f"Tag: {word}\nPages:\n" + "".join(f"- {t}: {first_sentence(summary_of.get(t))}\n" for t in ts)
        r = ask(model, REFERENT2.format(word=word), user, REFERENT_SCHEMA, max_tokens=60)
        referents.append((r or {}).get("referent", "?"))
    user = "".join(f"{i + 1}. {ref}\n" for i, ref in enumerate(referents))
    r = ask(model, COMPARE.format(word=word), user, COMPARE_SCHEMA)
    if r is None:
        return None
    used = {a["thing"] for a in r["assignments"]}
    return len(used) > 1, referents


# (d): one call, reasoning first, with prompt 8's definition of a meaning.
REASONED = """A tag of a personal archive of web pages is used on the groups of pages below. Decide whether the word has one meaning in all groups or names unrelated things.

- First, for each group, write refers_to: what the word itself refers to there, in a short phrase. A meaning is what the word itself refers to, not the subject of the pages: history pages about marathons, Kyoto and the French Revolution all use history in one meaning; food pages about bread, fruit and coffee all use food in one meaning.
- Then set same_meaning. It is false only when the word names things that nothing connects except the word: python the programming language and python the snake; java the programming language and java the island; mercury the planet and mercury the chemical element. Kinds, parts or aspects of one thing are one meaning: phones, cars and software are all technology.

Reply with a JSON object: {"groups": [{"group": 1, "refers_to": "..."}], "same_meaning": true}."""
REASONED_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["groups", "same_meaning"], "properties": {
    "groups": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["group", "refers_to"],
               "properties": {"group": {"type": "integer"}, "refers_to": {"type": "string"}}}},
    "same_meaning": {"type": "boolean"}}}


def reasoned(model, word, groups, think):
    r = ask(model, REASONED, groups_text(word, groups), REASONED_SCHEMA, think=think)
    return None if r is None else (not r["same_meaning"], [g["refers_to"] for g in r["groups"]])


COND = os.environ.get("COND", "a")
REPS = int(os.environ.get("REPS", "2"))
run = {
    "a": ("current prompt, qwen3:4b, thinking off", engine, "qwen3:4b", False),
    "b": ("staged, qwen3:4b", staged, "qwen3:4b", False),
    "c": ("staged, qwen3:14b", staged, "qwen3:14b", False),
    "d": ("reasoning first, qwen3:4b", reasoned, "qwen3:4b", False),
    "d14": ("reasoning first, qwen3:14b", reasoned, "qwen3:14b", False),
    "b2": ("staged with summaries, qwen3:4b", staged2, "qwen3:4b", False),
    "b2d": ("staged with summaries AND reasoning first agree, qwen3:4b", None, "qwen3:4b", False),
    "e": ("current prompt, qwen3:4b, thinking on", engine, "qwen3:4b", True),
}[COND]
name, fn, model, think = run
start = time.time()
rows = []
# Neutral examples, none of them test words.
NEUTRAL = os.environ.get("NEUTRAL") == "1"
if NEUTRAL:
    REFERENT2 = REFERENT2.replace('(for "history": "history in general"; for "music": "music in general")', '(for "furniture": "furniture in general"; for "vehicles": "vehicles in general")').replace(
        '"the programming language", "the snake", "the island", "the car maker", "food in general"', '"the flying mammal", "the baseball club", "the computer device", "clothing in general"')
    COMPARE = COMPARE.replace("phones, cars and software are all technology; bread, fruit and coffee are all food; jazz and the violin are both music", "chairs, tables and beds are all furniture; cars, bikes and ships are all vehicles; painting and sculpture are both art").replace(
        "python the programming language and python the snake; java the programming language and java the island; mercury the planet and mercury the chemical element", "bat the flying mammal and bat the baseball club; seal the animal and seal the stamp on a document; mouse the animal and mouse the computer device")
    REASONED = REASONED.replace("history pages about marathons, Kyoto and the French Revolution all use history in one meaning; food pages about bread, fruit and coffee all use food in one meaning", "furniture pages about chairs, tables and beds all use furniture in one meaning; art pages about painting and sculpture all use art in one meaning").replace(
        "python the programming language and python the snake; java the programming language and java the island; mercury the planet and mercury the chemical element", "bat the flying mammal and bat the baseball club; seal the animal and seal the stamp on a document; mouse the animal and mouse the computer device").replace(
        "phones, cars and software are all technology", "cars, bikes and ships are all vehicles")
    for prompt in (REFERENT2, COMPARE, REASONED):
        for w in ("history", "music", "food", "technology", "python", "java", "mercury"):
            assert f'"{w}' not in prompt and f" {w} " not in prompt, (w, prompt[:80])


def both(model, word, groups, think):
    b = staged2(model, word, groups, think)
    d = reasoned(model, word, groups, think)
    if b is None or d is None:
        return None
    return b[0] and d[0], (b[1], d[1])


fn = fn or both
for word, (truth, groups) in CASES.items():
    for _ in range(REPS):
        out = fn(model, word, groups, think)
        verdict, detail = (out if isinstance(out, tuple) else (out, None)) if out is not None else (None, None)
        rows.append((word, truth, verdict, detail))
elapsed = time.time() - start
right = sum(1 for _, t, v, _ in rows if v == t)
false_split = sorted({w for w, t, v, _ in rows if not t and v})
missed = sorted({w for w, t, v, _ in rows if t and v is False})
failed = sorted({w for w, t, v, _ in rows if v is None})
print(f"({COND}) {name}: right {right}/{len(rows)} · {elapsed / len(CASES) / REPS:.1f} s per word")
print(f"    broad words split: {', '.join(false_split) or '-'}")
print(f"    ambiguous words missed: {', '.join(missed) or '-'}")
print(f"    unusable replies: {', '.join(failed) or '-'}")
for w, t, v, d in rows[::REPS]:
    if d and v != t:
        print(f"    {w}: {d}")

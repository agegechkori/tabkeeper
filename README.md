# tabkeeper

Declare tab bankruptcy without losing anything. tabkeeper reads every open tab in your browsers, asks an LLM (cloud or local) to summarize and tag each page, and writes one Markdown note per tab, with hierarchical hashtags that stay consistent across thousands of pages.

**Status:** early development. See [docs/PLAN.md](docs/PLAN.md) for the design and roadmap. Reading tabs straight from browsers is not built yet; for now, tabkeeper works from a list of URLs.

## Usage

tabkeeper talks to any OpenAI-compatible chat server. By default it uses [Ollama](https://ollama.com) at `http://localhost:11434/v1` with the model `llama3.1:8b`, plus the embedding model `nomic-embed-text` to pick which existing tags the model sees for each page:

```sh
ollama pull nomic-embed-text
```

Without it tabkeeper still works, showing the model the most used tags instead, and says so when it starts.

```sh
cargo build --release

# urls.txt: one URL per line, optionally followed by a title; # starts a comment.
./target/release/tabkeeper import urls.txt --model qwen3:4b
```

With a thinking model such as `qwen3`, turn thinking off in the config file (see `tabkeeper config`); otherwise each page can take minutes instead of seconds:

```toml
[llm.extra_body]
reasoning_effort = "none"
```

This writes to `tabkeeper-out/` (change it with `--out`):

- `notes/*.md`: one note per page, with a title, the URL, a short summary and 3–5 tags such as `#technology/programming-languages/rust #technology/memory-safety`. The title is the page's own title (as in the browser tab) followed by the model's in parentheses, e.g. `Saffron - Wikipedia (Saffron: Spice from Saffron Crocus)`; `[notes] title` in the config picks one or the other instead. Pages without a summary still get a short stub note saying why, so no tab is lost: `#status/unreachable` when the page can't be loaded (404, dead domain, timeout, blocked), `#status/failed` when it loads but can't be summarized (a PDF, or the model's reply was unusable).
- `_tags.md`: all tags with page counts
- `_index.md`: all notes, each listed under its most used tag
- `_report.md`: the report of the latest run, including every page that failed or couldn't be loaded, and why
- `tabkeeper.db`: the SQLite database everything is rendered from

Several pages are processed at once (`[run]` in the config). Runs can be interrupted and restarted: finished pages are kept, and URLs already processed are skipped. If the model server stops responding, the run stops after a few retries and leaves the remaining pages for the next run. Other commands:

- `tabkeeper report`: print the report of the latest run again (`--json` for scripts)
- `tabkeeper revise-tags [--yes]`: review the tags (see below); `tabkeeper undo` reverts the last review
- `tabkeeper render`: rewrite the notes from the database
- `tabkeeper config`: show where the config file goes, with an example ([config.example.toml](config.example.toml))

Each run ends with a report: how many tabs were done, unreachable or failed (by cause), languages and domains, the model's token use, speed and cost, Ollama's memory and GPU use, the tags, and warnings about problems it noticed, such as a thinking model or pages cut short.

### Tag review

At the end of each import, tabkeeper reviews the tags with the whole archive in view. While pages are processed, each gets a few tags of its own, starting with its broad topic (`bread`, `investing`, `japan`). The review then builds one vocabulary for all pages and retags every page from it:

1. Pages are grouped by what they're about, and the model names each group with one tag that fits most of its pages. Tags already on several pages join the vocabulary too.
2. A one-word tag used for unrelated things gets a qualified tag per meaning: `python` becomes `python-programming` and `python-snake`. The model confirms each such case, with thinking allowed for those few requests (`think_meanings`).
3. Each page gets 1 to 3 tags from the vocabulary tags nearest to it. Tags that end up on fewer than 3 pages (`min_support`) are dropped, so every tag gathers several pages.

You see the vocabulary, each tag with its page count and description, and approve the retag in one step:

```
  15  technology            Technology companies, products and services
   7  python-programming    Python, the programming language
   5  python-snake          Pythons, the snakes
   ...
   1. retag  118 of 141 pages from these 32 tags (now 211 tags in use)
Apply them? [Y]es, [n]o, or the numbers to skip (e.g. 2,5):
```

Then it puts tags that aren't in the tag tree yet into it. The model first sorts them into broad domains (technology, science, food, …), then organizes each domain into categories, and you approve the result the same way, one line per category:

```
   1. place  science/chemistry ← rust-corrosion, oxidation (new: science, chemistry)
   2. place  technology/programming-languages ← rust, python, go (new: technology, programming-languages)
```

Notes then carry the tag's whole path (`#science/chemistry/rust-corrosion`), which Obsidian shows as nested tags, `_tags.md` shows the tree, and `_index.md` lists pages by top-level category. Set `style = "flat"` in `[tags]` for plain tags (`#rust-corrosion`) and no tree; switching back and forth is just `tabkeeper render`. The top-level categories come from your pages; to choose them yourself, list them in a file and pass `--taxonomy taxonomy.toml` (or set `taxonomy` in `[tags]`): see [taxonomy.example.toml](taxonomy.example.toml). With `strict = true` only those are allowed.

Words the model found to have one meaning, and placements you decline, are remembered and not asked about again. With `vocabulary = false` in `[reconcile]`, the review instead only merges duplicate tags and splits ambiguous ones, pair by pair. `--yes` applies everything without asking; without a terminal nothing is applied. `tabkeeper revise-tags` runs the review on its own, and `tabkeeper undo` reverts the last applied set of changes (run it twice to revert both the tree and the retag of one review); its changes then count as declined. The review can use its own model (`[reconcile]` in the config).

Before a big run, `--dry-run` shows what would be processed and estimates tokens, cost and time (from your last run, once there is one) without fetching or saving anything. `--max-tokens N` and `--max-cost DOLLARS` stop starting new pages once a limit would be reached; the rest stay pending for the next run. With a limit set, the tag review at the end of the run is skipped; run `tabkeeper revise-tags` when you're ready to spend on it. For cloud models, set `price_input_per_mtok` and `price_output_per_mtok` in `[llm]` to see costs.

Useful `import` options: `--lang de` or `--lang original` for summaries in another language (tags stay English), `--limit N` to process only N pages, `--concurrency N` for how many tabs are processed at once, `--max-tags N` for the most tags a page gets, `--retry-failed` to try failed and unreachable pages again, and `--deny RULE` / `--allow RULE` to skip or limit URLs for one run, e.g. `--deny domain:mail.google.com`. Rules are `domain:`, `glob:`, `regex:` or `prefix:`; permanent ones go in `[filter]` in the config.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

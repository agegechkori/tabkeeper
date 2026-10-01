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

- `notes/*.md`: one note per page, with a title, the URL, a short summary and 3–5 tags such as `#rust-programming #memory-safety`. The title is the page's own title (as in the browser tab) followed by the model's in parentheses, e.g. `Saffron - Wikipedia (Saffron: Spice from Saffron Crocus)`; `[notes] title` in the config picks one or the other instead. Pages without a summary still get a short stub note saying why, so no tab is lost: `#status/unreachable` when the page can't be loaded (404, dead domain, timeout, blocked), `#status/failed` when it loads but can't be summarized (a PDF, or the model's reply was unusable).
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

At the end of each import, tabkeeper reviews the tags: it merges tags that mean the same (`board-game` / `board-games`, `ml` / `machine-learning`) and splits a tag used for different meanings into one tag per meaning (`rust` for the language, `rust-corrosion` for the chemistry). Code finds the candidates, using the tag names and the pages' embeddings, and the model decides. You then see the proposed changes as a numbered list:

```
   1. merge  board-games → board-game (4 + 2 pages)
   2. split  rust → rust (3 pages), rust-corrosion (1 page)
Apply them? [Y]es, [n]o, or the numbers to skip (e.g. 2,5):
```

Changes you decline, and pairs the model kept apart, are remembered and not proposed again. `--yes` applies everything without asking; without a terminal nothing is applied. `tabkeeper revise-tags` runs the review on its own, and `tabkeeper undo` reverts the last applied review. The review can use its own model (`[reconcile]` in the config).

Before a big run, `--dry-run` shows what would be processed and estimates tokens, cost and time (from your last run, once there is one) without fetching or saving anything. `--max-tokens N` and `--max-cost DOLLARS` stop starting new pages once a limit would be reached; the rest stay pending for the next run. For cloud models, set `price_input_per_mtok` and `price_output_per_mtok` in `[llm]` to see costs.

Useful `import` options: `--lang de` or `--lang original` for summaries in another language (tags stay English), `--limit N` to process only N pages, `--concurrency N` for how many tabs are processed at once, `--max-tags N` for the most tags a page gets, `--retry-failed` to try failed and unreachable pages again, and `--deny RULE` / `--allow RULE` to skip or limit URLs for one run, e.g. `--deny domain:mail.google.com`. Rules are `domain:`, `glob:`, `regex:` or `prefix:`; permanent ones go in `[filter]` in the config.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

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

- `notes/*.md`: one note per page, with a title, the URL, a short summary and 3–5 tags such as `#rust-programming #memory-safety`. Pages without a summary still get a short stub note saying why, so no tab is lost: `#status/unreachable` when the page can't be loaded (404, dead domain, timeout, blocked), `#status/failed` when it loads but can't be summarized (a PDF, or the model's reply was unusable).
- `_tags.md`: all tags with page counts
- `_index.md`: all notes, each listed under its most used tag
- `tabkeeper.db`: the SQLite database everything is rendered from

Several pages are processed at once (`[run]` in the config). Runs can be interrupted and restarted: finished pages are kept, and URLs already processed are skipped. If the model server stops responding, the run stops after a few retries and leaves the remaining pages for the next run. Other commands:

- `tabkeeper report`: print the tag tree with page counts
- `tabkeeper render`: rewrite the notes from the database
- `tabkeeper config`: show where the config file goes, with an example ([config.example.toml](config.example.toml))

Useful `import` options: `--lang de` or `--lang original` for summaries in another language (tags stay English), `--limit N` to process only N pages, `--retry-failed` to try failed and unreachable pages again, and `--deny RULE` / `--allow RULE` to skip or limit URLs for one run, e.g. `--deny domain:mail.google.com`. Rules are `domain:`, `glob:`, `regex:` or `prefix:`; permanent ones go in `[filter]` in the config.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

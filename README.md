# tabkeeper

Declare tab bankruptcy without losing anything. tabkeeper reads every open tab in your browsers, asks an LLM (cloud or local) to summarize and tag each page, and writes one Markdown note per tab, with hierarchical hashtags that stay consistent across thousands of pages.

**Status:** early development. See [docs/PLAN.md](docs/PLAN.md) for the design and roadmap. Reading tabs straight from browsers is not built yet; for now, tabkeeper works from a list of URLs.

## Usage

tabkeeper talks to any OpenAI-compatible chat server. By default it uses [Ollama](https://ollama.com) at `http://localhost:11434/v1` with the model `llama3.1:8b`.

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

- `notes/*.md`: one note per page, with a title, the URL, a short summary and hierarchical tags such as `#technology/programming-languages/rust`
- `_tags.md`: the tag tree with page counts
- `_index.md`: all notes grouped by top-level tag
- `tabkeeper.db`: the SQLite database everything is rendered from

Runs can be interrupted and restarted: finished pages are kept, and URLs already processed are skipped. Other commands:

- `tabkeeper report`: print the tag tree with page counts
- `tabkeeper render`: rewrite the notes from the database
- `tabkeeper config`: show where the config file goes, with an example ([config.example.toml](config.example.toml))

Useful `import` options: `--lang de` or `--lang original` for summaries in another language (tags stay English), `--limit N` to process only N pages, and `--retry-failed`.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

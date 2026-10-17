# Working in this repository

## Keep command output small

Long sessions here run out of context mostly because of shell output. Keep every result to what you need to act on.

- **Read files with the Read tool,** using `offset`/`limit` for large ones (`src/pipeline.rs`, `src/main.rs` and `src/db/review.rs` are over 1,000 lines). Don't `cat` whole source files, logs or notes.
- **Find before reading:** `grep -n` (or the Grep tool) for the function or string, then read only those lines.
- **Filter command output** through `grep`, `head`, `tail` or `cut -c1-200`. Never print a whole log, database dump or note folder.
- **Cargo:**
  - `cargo test -q 2>&1 | grep -E "test result|panicked" -A6`: just the summary, plus failures if any.
  - `cargo clippy --all-targets -- -D warnings 2>&1 | grep -E "^(warning|error)" -A7 | head -30`
  - `cargo build --release -q`
- **Experiment runs** (`import`, `revise-tags` on test archives) print hundreds of lines. Redirect them to a log file, then grep it for what matters, e.g. `grep -E "^Vocabulary|fit none|retag|Tag review" run.log`.
- **SQLite:** select only the needed columns, and add `LIMIT`.
- **Comparisons:** `python3 scripts/compare_trees.py --clusters <urls> <dir> | grep -E "average cluster|two meanings|distinct"` for the headline numbers.
- **Editing:** after a `cargo fmt`, a multi-line edit anchor may have been reformatted. Check it with `grep -n ... -A5` before editing, instead of rereading the file.

## Project notes

- **Language and model:** Rust (edition 2024). The default model is local `qwen3:4b` on Ollama, with thinking off.
- **Tests and lint:** run `cargo test` and `cargo clippy --all-targets -- -D warnings`. Both must pass before any commit.
- **Experiments:** they live in `target/experiments/` (ignored by git).
  - The imported test archive is `target/experiments/5a/p`. Copy it (`cp -R p <new>`) before running a review on it, because reviews remember their verdicts in the database.
  - Use `scripts/cluster_urls.txt` (141 pages in 27 topic clusters) with `scripts/compare_trees.py --clusters` to measure tag quality.
  - Use `scripts/meanings_bench.py` to benchmark the "several meanings?" check.
- **Docs:** the design and phase plan are in `docs/PLAN.md`.

# tabkeeper — plan

A Rust CLI for declaring tab bankruptcy. It reads every open tab in the major browsers (thousands of them), asks an LLM to summarize and tag each page, and writes one Markdown note per tab. Tags form a hierarchy; the tool keeps them consistent and revises them at the end.

## Requirements

- **Browsers:** Chrome, Edge, Brave, Vivaldi, Opera, Firefox, and Safari (macOS). All windows and profiles are included; `--browser` and `--profile` filter them.
- **Operating systems:** macOS, Linux, Windows.
- **Scale:** thousands of tabs. Runs can resume after a crash, re-runs skip pages already done, and costs are estimated up front and can be capped.
- **Output, one Markdown note per tab:**
  - a title that briefly describes the page
  - the URL
  - a summary of a few sentences
  - hierarchical hashtags
- **LLM:** configurable, including local models. Cloud models can be sent just the URL instead of the page content. Responses follow a JSON schema and are validated.
- **Tags:** the tool reuses existing tags wherever possible, merges duplicates, and revises the hierarchy after the run. It ends by printing a tag tree with page counts.
- **Domain/URL filter:** a configurable allow list or deny list.
- **Language:** English by default. `--lang <code>` forces another language; `--lang original` writes each summary in the page's own language. Tags are always English.
- **Read-only by default:** the core tool never modifies browser data. An optional step can close processed tabs (see phase 7).

## Architecture

```
sources ─► filter/dedupe ─► fetch+extract ─► LLM ─► tag resolution ─► SQLite
                                                                        │
                                  revise-tags (merge/move/rename/split) ◄┤
                                                                        ▼
                                   render: notes/*.md + _tags.md + _index.md
```

SQLite is the source of truth. The Markdown files are always regenerated from it, so fixing a tag is just a database change followed by a re-render.

### Tab sources
- **Firefox:** reads `sessionstore-backups/recovery.jsonlz4` (the `mozlz4` format, decoded with `lz4_flex`).
- **Chromium family:** parses the SNSS files in `Sessions/`.
- **Safari, and Chromium browsers on macOS:** AppleScript/JXA via `osascript`.
- **Import:** `import <urls.txt | bookmarks.html>`.
- **Optional, later:** a companion WebExtension that talks to the CLI over native messaging, for pages behind a login.

With thousands of tabs, most are discarded or unloaded by the browser, so page content is fetched over HTTP.

### Filter and dedupe
- Browser-internal and local schemes are always skipped: `chrome://`, `edge://`, `about:`, `file://`, extension pages.
- `[filter] mode = "allow" | "deny"`. Rules can be `domain:` (subdomains included), `glob:`, `regex:` or `prefix:`. `--allow` and `--deny` flags add rules for a single run.
- URLs are normalised before deduplication: tracking parameters (`utm_*`, `fbclid`, …) and the fragment are removed.

### Fetch and extract
- `reqwest` with a limit on concurrent requests per domain, timeouts, and retries with backoff.
- Readability-style extraction with `dom_smoothie`; PDFs with `pdf-extract`.
- Text is cut down to fit the configured token budget.
- Language is detected with `whatlang`.
- **Unreachable pages** (404s, dead domains, login walls, bot blocks) still get a stub note built from the browser's tab title, tagged `#status/unreachable`.

### LLM layer
- Thin adapters built directly on `reqwest`, one per protocol:
  - **OpenAI-compatible:** covers Ollama, LM Studio, llama.cpp, vLLM, OpenRouter and OpenAI.
  - **Anthropic.**
  - **Gemini.**
- **Structured output:** the JSON schema is generated from Rust types with `schemars` and passed to the provider's JSON-schema mode or tool calling. Responses are validated with `serde`, and invalid ones are retried once with the error included.
- **URL-only mode:** each provider has a capability flag saying whether its model can fetch pages itself (Anthropic web fetch, Gemini URL context, OpenAI web search). If the model can't fetch the page, the tool automatically sends the content instead.
- **Prompt layout:** instructions and the tag tree come first in the prompt, so providers that cache repeated prompt prefixes charge less for them.
- **`--batch` mode:** uses the Anthropic and OpenAI batch APIs, which cost about 50% less and return results asynchronously.
- **Throughput and cost controls:**
  - a global concurrency limit, with rate limiting via `governor` and `Retry-After` handling
  - `--max-cost` / `--max-tokens` caps
  - `--dry-run` prints the tab count, filter results and a cost estimate without calling the model

Response schema:
```json
{ "title": "...", "summary": "...", "language": "en",
  "tags": ["technology/programming-languages/rust"],
  "new_tags": [{"path": "technology/programming-languages/zig", "description": "..."}] }
```

### Tags

**Hierarchy**
- A tag is identified by its **full path**, so ambiguity is resolved by where a tag sits: `technology/programming-languages/rust` and `science/chemistry/rust` are different tags.
- Names are kebab-case, with at most 3 levels (configurable).
- Each tag has one parent; a page can carry several tags.
- A page may be tagged with an inner node when nothing more specific fits.
- Notes contain the full path (`#technology/programming-languages/rust`), which works as nested tags in Obsidian.
- Tags under `status/` are reserved for the tool itself, e.g. `status/unreachable`.

**Top level**
- By default the top-level categories emerge from the pages.
- `--taxonomy taxonomy.toml` supplies your own. It's a seed by default; with `strict = true`, only the listed top-level categories are allowed.

**What the model sees for each page**
- The top of the tree plus the branches most relevant to the page, with counts and descriptions.
- The model must reuse an existing path or propose a new one with a description.

**Bootstrap:** the first ~50 pages run one at a time so the tree settles; after that, pages run in parallel.

**Storage**
```sql
tags(id, name, parent_id, description, locked, UNIQUE(parent_id, name))
page_tags(page_id, raw_tag, resolved_tag_id)   -- raw_tag is never modified
tag_aliases(alias, tag_id, source, locked)      -- source: rule | llm | user
page_tag_overrides(page_id, raw_tag, tag_id)    -- per-page reassignments
revision_log(run_id, op, payload, approved_at)  -- audit trail and undo
```
Resolution order: per-page override, then alias, then the raw tag. Since the raw tags are never modified, undoing a revision means deleting its rows and re-rendering. Entries marked `locked` (your manual decisions) are never overridden by the LLM.

**Revision pass.** It runs at checkpoints (after 200 pages, then every ~1,000) and once at the end.
1. Code normalises tags (case, plurals, punctuation) and writes these as `rule` aliases.
2. The LLM gets the whole tree, alphabetised, with counts, plus example page titles for tags that could be ambiguous. It returns only the changes it proposes: **merge**, **move** (reparent), **create parent**, **rename**, and **reassign** (moves specific pages to another tag, decided from their stored title and summary).
3. Code validates the proposal:
   - chains are resolved to the final tag and cycles are rejected
   - a tag can't be sent to two different targets
   - new paths must be explicitly marked as new
   - the depth limit and locked entries are enforced
4. The changes are shown as a diff with the number of pages affected. You approve them, or `--yes` skips approval.
5. Approved changes are stored, logged, and the affected notes are re-rendered.
6. If the tree doesn't fit the model's context window, it is revised one top-level branch at a time.

**Final report:** the tag tree with two counts per tag, pages tagged with that tag exactly and pages anywhere under it. It's printed and saved as `_tags.md`.

### Output
```
tabkeeper-out/
  tabkeeper.db
  notes/<slug-of-title>.md
  _tags.md
  _index.md        # all notes, grouped by top-level tag
```
Note format (Obsidian-compatible):
```markdown
---
url: https://example.com/article
browser: firefox
captured: 2026-09-30T14:02:00Z
lang: en
tags: [technology/ai/llm, technology/hardware/gpu]
---
# Fine-tuning small LLMs on consumer GPUs

**URL:** https://example.com/article

A few sentences of summary…

#technology/ai/llm #technology/hardware/gpu
```

### CLI
```
tabkeeper sources                    # browsers/profiles found, tab counts
tabkeeper run [--dry-run] [--batch] [--lang en|<code>|original] [--taxonomy f] [--allow/--deny r] [--max-cost n]
tabkeeper import <file>
tabkeeper revise-tags [--yes]
tabkeeper undo <run-id>
tabkeeper render
tabkeeper report
tabkeeper close-tabs                 # optional, phase 7
```
- Config: `config.toml` in the platform's config directory (via `directories`).
- API keys: taken from environment variables named in the config.

### Crates
tokio, reqwest, clap, serde, schemars, rusqlite (bundled), lz4_flex, dom_smoothie, pdf-extract, whatlang, globset, regex, governor, indicatif, directories, toml, url.

## Phases
1. **Core:** config, SQLite schema, `import` from a URL list, fetch and extract, OpenAI-compatible/Ollama adapter, hierarchical tag registry, Markdown rendering, report, `_index.md`.
2. **Scale:** concurrency, rate limits, resume, dedupe, dry-run cost estimate, budget caps, progress bar, unreachable-page stubs.
3. **Browser sources:** Firefox; the Chromium family via SNSS; Safari and macOS via AppleScript; profile discovery on all three operating systems.
4. **Tag revision:** checkpoints, validation, approval diff, revision log, undo, taxonomy file.
5. **Providers:** Anthropic and Gemini adapters, URL-only mode, `--batch`.
6. **Distribution:** CI builds for macOS, Linux and Windows; GitHub Releases; `cargo install`.
7. **Optional:** companion extension (pages behind a login) and `close-tabs`. Closing tabs is only possible via the extension or AppleScript, is off by default, and asks for confirmation.

License: MIT OR Apache-2.0.

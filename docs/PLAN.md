# tabkeeper — plan

A Rust CLI for declaring tab bankruptcy. It reads every open tab in the major browsers (thousands of them), asks an LLM to summarize and tag each page, and writes one Markdown note per tab. Pages get simple flat tags while they are processed; at the end of each run, one reconciliation pass cleans the tags up and, if you want, organizes them into a hierarchy.

**Status:** phase 1 (the core pipeline, working from a URL list) is done ([PR #1](https://github.com/agegechkori/tabkeeper/pull/1)), and so are phase 2a, flat tags per page with embeddings ([PR #3](https://github.com/agegechkori/tabkeeper/pull/3)), and phase 2b, processing at scale ([PR #4](https://github.com/agegechkori/tabkeeper/pull/4)), followed by fixes from a review of the whole codebase ([PR #5](https://github.com/agegechkori/tabkeeper/pull/5)). Phase 2c, the report, the dry run and budgets, is in progress.

## Requirements

- **Browsers:** Chrome, Edge, Brave, Vivaldi, Opera, Firefox, and Safari (macOS). All windows and profiles are included; `--browser` and `--profile` filter them.
- **Operating systems:** macOS, Linux, Windows.
- **Scale:** thousands of tabs. Runs can resume after a crash, re-runs skip pages already done, and costs are estimated up front and can be capped.
- **Output, one Markdown note per tab:**
  - a title that briefly describes the page
  - the URL
  - a summary of a few sentences
  - hashtags, either hierarchical (`#technology/programming-languages/rust`) or flat (`#rust-programming`), as configured
- **LLM:** configurable, including local models. Cloud models can be sent just the URL instead of the page content. Responses follow a JSON schema and are validated.
- **Tags:** the tool reuses existing tags wherever possible, and reconciles them at the end of each run: synonyms merged, ambiguous tags split, and, in hierarchical mode, tags organized into a tree.
- **Final report:** tabs processed, failures by cause, model usage and speed, and all tags with page counts. Printed and saved as `_report.md`.
- **Domain/URL filter:** a configurable allow list or deny list.
- **Language:** English by default. `--lang <code>` forces another language; `--lang original` writes each summary in the page's own language. Tags are always English.
- **Read-only by default:** the core tool never modifies browser data. An optional step can close processed tabs (see phase 7).

## Architecture

```
sources ─► filter/dedupe ─► fetch+extract ─► LLM: title, summary, flat tags ─► SQLite
                                  ▲                                           │
                   embeddings: pick relevant existing tags                    │
                                                                              ▼
                     end of run: reconcile tags (merge, split, build tree if hierarchical)
                                                                              │
                                                              you approve the changes
                                                                              ▼
                          render: notes/*.md + _tags.md + _index.md + _report.md
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
- Text is cut down to fit the configured budget (`llm.max_input_chars`).
- Language is detected with `whatlang`.
- **Every tab gets a note.** Unreachable pages (404s, dead domains, login walls, bot blocks) get a stub note built from the browser's tab title, tagged `#status/unreachable`; pages that load but can't be summarized (PDFs until they're supported, invalid model replies, prompts the server rejects) get one tagged `#status/failed`. Each stub says why.

### LLM layer
- Thin adapters built directly on `reqwest`, one per protocol:
  - **OpenAI-compatible:** covers Ollama, LM Studio, llama.cpp, vLLM, OpenRouter and OpenAI.
  - **Anthropic.**
  - **Gemini.**
- **Structured output:** a hand-written JSON schema, passed to the provider's JSON-schema mode or tool calling. Responses are validated with `serde`, and invalid ones are retried once with the error included.
- **Server-specific options:** `[llm.extra_body]` merges extra fields into every request. On Ollama, `reasoning_effort = "none"` turns off a thinking model's reasoning; in testing, that took `qwen3:4b` from about 3 minutes to about 10 seconds per page.
- **Separate models per stage:** per-page summaries, embeddings and the end-of-run tag reconciliation can each use a different model, e.g. a fast model for pages and a stronger one, or thinking turned on, for the one-time reconciliation.
- **URL-only mode:** each provider has a capability flag saying whether its model can fetch pages itself (Anthropic web fetch, Gemini URL context, OpenAI web search). If the model can't fetch the page, the tool automatically sends the content instead.
- **Prompt layout:** instructions and the tag vocabulary come first in the prompt, so providers that cache repeated prompt prefixes charge less for them.
- **`--batch` mode:** uses the Anthropic and OpenAI batch APIs, which cost about 50% less and return results asynchronously.
- **Throughput and cost controls:**
  - a global concurrency limit, with rate limiting via `governor` and `Retry-After` handling
  - `--max-cost` / `--max-tokens` caps
  - `--dry-run` prints the tab count, filter results and a cost estimate without calling the model
- **Page-level vs. run-level errors:** a page the model can't handle (invalid reply twice, too long) is recorded as failed and the run moves on; a server problem (connection refused, authentication, server error) stops the run and leaves the remaining pages pending.

Per-page response schema:
```json
{ "title": "...", "summary": "...", "language": "en",
  "tags": ["rust", "memory-safety"],
  "new_tags": [{"name": "memory-safety", "description": "..."}] }
```

### Embeddings
Served through the same OpenAI-compatible server's `/v1/embeddings` endpoint, with its own model setting (e.g. `nomic-embed-text` on Ollama, or `bge-m3` when summaries are in many languages). Vectors are stored in SQLite; a brute-force similarity search is fast enough at this scale. No machine-learning runtime is bundled into the binary.

Embeddings find candidates; the LLM makes the decisions. They are used for:
1. **Choosing which existing tags to show the model** for each page: the most similar to the page, instead of the most used. This is what keeps tag reuse working with thousands of tags.
2. **Finding likely duplicate tags** (`ml` / `machine-learning`) for the reconciliation pass to confirm. Related-but-different tags (`react` / `react-native`) also score as similar, so the LLM confirms every merge.
3. **Splitting ambiguous tags:** clustering the summaries of the pages that share a tag, e.g. all `rust` pages, shows whether it covers more than one meaning.
4. **A first draft of the tree** in hierarchical mode: clusters of similar tags, which the LLM then names and corrects.
5. **Spotting the same article** saved under different URLs.

### Tags

**While pages are processed: flat tags**
- Each page gets 3–5 flat, lowercase kebab-case English tags. Choosing a few topic tags is an easy task that small, fast models do well.
- The model sees the existing tags most relevant to the page (picked with embeddings), with counts and descriptions, and must reuse one when it fits.
- Case, spacing, punctuation and missing hyphens are normalized in code as each page is processed (`Machine Learning`, `machinelearning` → `machine-learning`), and recorded as `rule` aliases. Plural and singular forms are not merged here, because the plural can mean something else (`glasses` is not `glass`, `windows` is not `window`); reconciliation proposes those merges for confirmation.
- Tags under `status/` are reserved for the tool itself, e.g. `status/unreachable`.
- Pages don't depend on each other's tags beyond the shared vocabulary, so they can be processed in parallel.

**At the end of each run: reconciliation**

It runs once at the end of every run, including a run that stopped early, and on demand with `tabkeeper revise-tags`. There are no checkpoints during the run: with flat tags and embedding-picked vocabulary there's little to fix mid-run, and one pass over all the tags gives a better result. If long runs turn out to fragment the vocabulary, a cheap synonym merge during the run can be added later.

1. Code proposes the mechanical merges, such as plural and singular forms (`board-games` / `board-game`), for the LLM to confirm with the rest.
2. Embeddings propose candidates: likely duplicates, tags whose pages fall into separate clusters, and (hierarchical mode) groups of related tags.
3. The LLM decides, returning only the changes:
   - **merge:** synonyms and variants become one tag.
   - **split:** an ambiguous tag becomes several, and each of its pages is reassigned based on its stored title and summary. In flat mode the names carry the meaning (`rust-programming`, `rust-corrosion`); in hierarchical mode the position does (`technology/programming-languages/rust`, `science/chemistry/rust`).
   - **place / move** (hierarchical mode only): put each tag in the tree, creating parent categories as needed.
   - **rename.**
4. Code validates the proposal:
   - chains are resolved to the final tag and cycles are rejected
   - a tag can't be sent to two different targets
   - new names must be explicitly marked as new
   - the depth limit and locked entries are enforced
5. The changes are shown as a list with the number of pages each affects. You approve them, or `--yes` skips approval.
6. Approved changes are stored, logged, and the affected notes are re-rendered.
7. If the vocabulary doesn't fit the model's context window, it is reconciled in chunks (one top-level branch, or one cluster, at a time).

**Stability across runs.** The first reconciliation builds the structure. Later runs place new tags into it and propose restructuring as changes you approve, so notes don't get different tags every run. Your manual decisions are `locked` and never overridden by the LLM.

**Flat or hierarchical: `[tags] style = "hierarchical" | "flat"`**
- Default: `hierarchical`.
- Every tag has a unique flat name either way, qualified when the plain word is ambiguous (`rust-programming`). The tree is an optional layer on top.
- Switching from hierarchical to flat is just a re-render (`tabkeeper render`), with no LLM calls. Switching from flat to hierarchical needs the tree built once (`tabkeeper revise-tags`).
- Flat mode skips building the tree, so its reconciliation is cheaper.

**Hierarchy details** (hierarchical mode)
- Tags have at most 3 levels (configurable); each tag has one parent; a page can carry several tags.
- A page may be tagged with an inner node when nothing more specific fits.
- Notes contain the full path (`#technology/programming-languages/rust`), which works as nested tags in Obsidian.
- The top-level categories emerge from your pages by default. `--taxonomy taxonomy.toml` supplies your own: a seed by default, or with `strict = true`, the only top-level categories allowed.

**Storage**
```sql
tags(id, name UNIQUE, parent_id, path, description, locked)  -- parent_id/path only used in hierarchical mode
page_tags(page_id, raw_tag, resolved_tag_id)   -- raw_tag is exactly what the model returned, never modified
tag_aliases(alias, tag_id, source, locked)      -- merges; source: rule | llm | user
page_tag_overrides(page_id, raw_tag, tag_id)    -- per-page reassignments from splits
revision_log(run_id, op, payload, approved_at)  -- audit trail and undo
embeddings(kind, ref_id, model, vector)         -- tags and page summaries
```
Resolution order: per-page override, then alias, then the raw tag. Since the raw tags are never modified, undoing a revision means deleting its rows and re-rendering.

### Final report
Printed at the end of every run and saved as `_report.md` in the output folder, one per run. For resumed runs it shows this run's numbers alongside the totals for the whole archive. `--json` prints it as JSON for scripting. The full list of failed URLs with their errors goes into `_report.md`, not the terminal.

```
tabkeeper run 2026-09-30 14:02, finished in 1h 42m

Tabs
  Found            4,812   (Chrome 3,120 · Firefox 1,540 · Safari 152)
  Duplicates         611   merged into existing notes
  Filtered out       203   (deny list 181 · non-web 22)
  Already done       950   from earlier runs, skipped
  Processed        3,048
    Done           2,871
    Unreachable      142   stub notes tagged #status/unreachable
      404/410         71 · DNS/connection 38 · timeout 19 · blocked (403/429) 14
    Failed            35   stub notes tagged #status/failed
      PDF 21 · invalid model reply 9 · too long for model 5
  Pending              0

Languages  en 2,410 · de 301 · fr 88 · ja 45 · other 27
Domains    github.com 412 · youtube.com 233 · en.wikipedia.org 190 · ...

Model
  Summaries   qwen3:4b · Ollama at localhost:11434 · reasoning_effort=none
              7.6 GB loaded, 100% on GPU · context 32k tokens
  Embeddings  nomic-embed-text · 4,015 vectors in 41 s
  Tag review  qwen3:4b with thinking · 14 requests · 6m 12s

  Requests    3,112 summary requests · 64 retries (invalid replies)
  Tokens      9.8M input · 0.9M output · 3,210 input / 295 output per page
  Speed       2.1 s per page (4 at a time) · 140 output tokens/s
  Time        a summary request took 7.9 s, including any wait in the server's queue · a fetch 0.4 s
  Cost        $0.00 (local)
  Page limits 412 pages cut to fit max_input_chars (12,000)
              5 rejected as too long for the model
              9 failed after two invalid replies
  URL-only    0 pages (the model can't fetch pages itself)

  Warnings    none
tabkeeper     CPU 38 s · peak memory 64 MB

Tags       1,204 tags (318 new this run) · 12 untagged pages
Revision   87 merges · 9 splits (e.g. rust → rust-programming, rust-corrosion)
           · 41 tags used only once

<tag tree (hierarchical) or tag list (flat), with page counts>

Notes written to tabkeeper-out/ (2,871 notes + 177 stubs)
```

Where the model numbers come from:

| Line | Source | Availability |
|---|---|---|
| Model, server, extra options | the config | always |
| Memory, GPU share, context window | Ollama's `/api/ps` | Ollama only |
| Requests, retries, invalid replies | counted by tabkeeper | always |
| Tokens | `usage` field in each response | when the server reports it; otherwise "not reported" |
| Speed | pages and output tokens ÷ the run's wall time; request time measured by tabkeeper | always. The OpenAI-compatible API gives only a total time per request, which includes time queued at the server when several pages run at once, so it can't be split into input and output speed |
| Model vs. fetch time per page | measured by tabkeeper | always |
| Cost | tokens × prices set in the config | cloud models with prices set |
| Page limits, URL-only | counted by tabkeeper | always / when used |
| Embeddings, tag review | same as above, per stage | when those stages ran |
| tabkeeper CPU and memory | the operating system (`getrusage`; a small crate on Windows) | always |

The model server's own CPU and RAM are not measured: it's a separate process, sometimes on another machine, and on a Mac most of its work is on the GPU, which can't be read without admin rights. Throughput and Ollama's `/api/ps` cover what matters.

Tokens and cost are shown per stage (summaries, embeddings, tag review) and in total.

**Warnings**, printed only when a problem is detected:
- **Thinking left on:** "Average 2,100 output tokens per page; the model may be thinking. On Ollama, set `reasoning_effort = "none"`."
- **Model not fully on the GPU:** "Only 60% of the model is on the GPU; a smaller model or a shorter context would be much faster."
- **Pages being cut:** "30% of pages were cut to fit `max_input_chars`; consider raising it if your model's context allows."
- **Model context too small:** "Prompts average 7,900 tokens but the model's context is 8k; some pages may be cut off by the server without an error."
- **Many retries:** "12% of replies were invalid JSON; try `structured_output = "json_schema"` or a larger model."

### Output
```
tabkeeper-out/
  tabkeeper.db
  notes/<slug-of-title>.md
  _tags.md         # tag tree or tag list, with page counts
  _index.md        # all notes, grouped by top-level tag (hierarchical) or by most-used tags (flat)
  _report.md       # the final report of the latest run, with the full list of failed URLs
```
Note format (Obsidian-compatible, hierarchical style):
```markdown
---
url: "https://example.com/article"
title: "Fine-tuning LLMs on a budget | Blog (Fine-tuning small LLMs on consumer GPUs)"
page_title: "Fine-tuning LLMs on a budget | Blog"
summary_title: "Fine-tuning small LLMs on consumer GPUs"
source: firefox
captured: 2026-09-30T14:02:00Z
lang: "en"
tags:
  - technology/ai/llm
  - technology/hardware/gpu
---

# Fine-tuning LLMs on a budget | Blog (Fine-tuning small LLMs on consumer GPUs)

**URL:** <https://example.com/article>

A few sentences of summary…

#technology/ai/llm #technology/hardware/gpu
```

### CLI
```
tabkeeper sources                    # browsers/profiles found, tab counts
tabkeeper run [--dry-run] [--batch] [--lang en|<code>|original] [--taxonomy f] [--allow/--deny r] [--max-cost n] [--json]
tabkeeper import <file>
tabkeeper revise-tags [--yes]
tabkeeper undo                       # reverts the last applied tag review
tabkeeper render
tabkeeper report [--json]
tabkeeper config
tabkeeper close-tabs                 # optional, phase 7
```
- Config: `config.toml` in the platform's config directory (via `directories`); `tabkeeper config` shows where, with an example.
- API keys: taken from environment variables named in the config.

### Crates
tokio, reqwest, clap, serde, serde_json, rusqlite (bundled), lz4_flex, dom_smoothie, pdf-extract, whatlang, globset, regex, governor, indicatif, directories, toml, url, chrono.

## Phases
1. **Core** ([PR #1](https://github.com/agegechkori/tabkeeper/pull/1)): config, SQLite schema, `import` from a URL list, fetch and extract, OpenAI-compatible/Ollama adapter, tag registry, Markdown rendering, report, `_index.md`.
2. **Flat tagging and scale**, in three pull requests:
   - 2a ([PR #3](https://github.com/agegechkori/tabkeeper/pull/3)): per-page flat tags, with embeddings choosing which existing tags the model sees
   - 2b ([PR #4](https://github.com/agegechkori/tabkeeper/pull/4)): concurrency, retries and rate limits, progress bar, unreachable-page stubs, failure causes, domain filter
   - 2c ([PR #6](https://github.com/agegechkori/tabkeeper/pull/6)): token usage per request, then dry-run cost estimate, budget caps, and the final report (everything except per-browser counts and the reconciliation numbers)
3. **Tag reconciliation**, in three pull requests:
   - 3a: the review for flat tags. Merge candidates come from plural forms and from embeddings of the tag names alone (tag descriptions made tags from the same page look alike); split candidates are tags whose pages form two dissimilar groups. The model decides in batches, the changes are listed for approval, applied in one transaction and logged with an undo record (`revisions`), and declined changes are remembered (`tag_decisions`). Instead of the alias and override tables planned above, a revision retargets `page_tags.resolved_tag_id` (the raw tags stay untouched) and stores what it changed.
   - 3b: the tree in hierarchical mode, `tags.style`, the taxonomy file. Placement runs after merges and splits, as its own approval and revision, so it sees the final tags. It is two steps, which small models handle much better than placing a mixed list in one go: each tag's broad domain, a few tags at a time, then one domain at a time, with all of that domain's tags in view, the categories inside it. Tag names stay unique and flat, so a category name is in one place in the tree; a path that runs into a category elsewhere is cut short there. Tags are placed once; moving placed tags around (restructuring) is left for later.
   - 3c: compare with phase 1's direct hierarchical tagging on the same 50–100 URLs: tree depth, tag reuse, whether `rust` is split correctly, speed.
4. **Browser sources:** Firefox; the Chromium family via SNSS; Safari and macOS via AppleScript; profile discovery on all three operating systems.
5. **Providers:** Anthropic and Gemini adapters, URL-only mode, `--batch`.
6. **Distribution:** CI builds for macOS, Linux and Windows; GitHub Releases; `cargo install`.
7. **Optional:** companion extension (pages behind a login) and `close-tabs`. Closing tabs is only possible via the extension or AppleScript, is off by default, and asks for confirmation.

Tag reconciliation now comes before browser sources, because it changes how pages are tagged and is what the quality of the output depends on.

License: MIT OR Apache-2.0.

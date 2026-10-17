# Tab Organizer v1 — Agent Implementation Plan

## 0. Mission

Build a fully functional **browser-independent tab-organization prototype** for macOS that uses a single WXT extension codebase and browser-specific builds only where the WebExtensions platform requires them. The prototype should target Chromium-based browsers and Firefox first, while keeping Safari support architecturally possible. It should also support an externally supplied predefined taxonomy.

1. Captures tabs from a browser extension built with **WXT**, using WXT's cross-browser `browser` abstraction rather than direct Chrome/Firefox globals.
2. Sends tab content and metadata to a local Rust application server.
3. Persists run state so long jobs are resumable and idempotent.
4. Uses a locally running `mistral.rs` HTTP API for summarization, topic extraction, tag reasoning, and embeddings.
5. Builds a reusable corpus-wide tag vocabulary instead of generating independent ad-hoc tags per page.
6. Produces Markdown notes suitable for an Obsidian-like PKMS.
7. Accepts an optional **external predefined taxonomy** from a local YAML/JSON file and can operate in auto, predefined-preferred, or predefined-strict tagging modes.
8. Lets the browser extension choose the processing scope: everything, selected windows, or selected tabs.
9. Optionally closes successfully processed tabs.
10. Supports two independent blacklists:
   - processing blacklist: tabs/domains that must never be processed;
   - close blacklist: tabs/domains that may be processed but must never be closed automatically.
11. Never loses progress if the app, container, model server, or machine is interrupted.

This is intentionally **v1 and tabs-only**. Do not spend implementation effort on the generalized document/PKMS domain architecture discussed for the future product. However, keep the code organized enough that future versions can generalize the concepts without rewriting the pipeline.

---

# 1. Product scope and assumptions

## 1.1 Target environment

Primary development and first supported environment:

- macOS on Apple Silicon.
- User's representative hardware: M1 Pro, 32 GiB unified memory.
- Browser extension built with WXT.
- Primary v1 browser targets: Chrome/Chromium and Firefox. Edge should build from the same source tree. Safari should remain architecturally possible, but Safari packaging is not a v1 release gate.
- Do not use direct `chrome.*` APIs in extension business logic. Use WXT's unified `browser` API. Browser-specific differences must be isolated in WXT configuration or small adapters.
- Rust application server.
- Application server runs in Docker during the prototype phase.
- `mistral.rs` runs natively on macOS, outside Docker, to retain Apple Silicon/Metal acceleration.
- Browser extension talks to the local application server over localhost HTTP. The endpoint/port is configurable and the extension uses the appropriate host permissions for each browser target.
- Application server talks to `mistral.rs` over HTTP.

Docker Desktop for Mac supports `host.docker.internal` for reaching host services from a container, and modern Docker Desktop can optionally use host networking. For v1, prefer the least surprising topology that keeps the application server localhost-only from the host's perspective. Do not expose the application server or model server to the LAN unless explicitly configured.

`mistral.rs` currently exposes OpenAI-compatible HTTP endpoints including chat completions and embeddings, and supports structured output. This makes it a good fit for the LLM-provider boundary in this prototype. WXT provides a unified browser-extension API and target-specific builds from one source tree.

## 1.2 Non-goals for v1

Do not implement:

- General PKMS connectors other than Markdown-file output.
- General arbitrary-document ingestion.
- Notion/Logseq/etc. integrations.
- A sophisticated desktop application.
- A taxonomy marketplace/registry.
- Remote taxonomy URLs as a required v1 feature; local YAML/JSON taxonomy files are sufficient for v1.
- Cloud LLM providers as a primary feature.
- Agentic tool use.
- Multi-user authentication/authorization.
- A remote/network deployment model.
- A sophisticated graph database.
- Fine-tuning.
- Perfect extraction from every possible web application.

The v1 product should be useful and robust for browser tabs first.

---

# 2. High-level architecture

```text
                         Chrome browser
                              |
                       Chrome extension
                              |
                         HTTP localhost
                              |
                              v
                 +---------------------------+
                 |   Tab Organizer Server    |
                 |          Rust             |
                 |                           |
                 | run manager               |
                 | tab state machine         |
                 | content processing        |
                 | tagging pipeline          |
                 | Markdown writer            |
                 | persistent local state     |
                 +-------------+-------------+
                               |
                         HTTP localhost/host
                               |
                               v
                 +---------------------------+
                 |        mistral.rs          |
                 |       native macOS         |
                 |                           |
                 | Qwen3-14B (text)          |
                 | Qwen3-Embedding-0.6B      |
                 +---------------------------+
```

Recommended v1 process topology:

```text
Chrome extension
      |
      | http://127.0.0.1:<server-port>
      v
Docker container: tab-organizer-server
      |
      | http://<host-service>:<llm-port>/v1
      v
Native mistral.rs process
```

For Docker Desktop on macOS, evaluate `host.docker.internal` first. If host networking is enabled and is simpler/reliable for the development environment, it may be used. The implementation must make the model endpoint configurable via environment/config rather than hardcoding the topology.

---

# 3. Core implementation principles

## 3.1 Persistence is an adapter

Do NOT let SQLite types, SQL strings, table names, or ORM types leak into the application/business logic.

Even though SQLite is the preferred v1 persistence implementation, define repository interfaces and implement SQLite behind them.

The prototype may be structurally simple, but maintain the following direction:

```text
HTTP / Extension
       |
Application logic
       |
Repository interfaces
       |
SQLite adapter
```

The application should remain testable with in-memory/fake repositories.

## 3.2 LLM is a provider

Do not embed direct `mistral.rs` request logic throughout the tagging algorithm.

Define an application-level provider interface such as:

```rust
trait LlmProvider {
    async fn generate(&self, request: LlmRequest) -> Result<LlmResponse>;
    async fn generate_structured<T>(&self, request: StructuredRequest) -> Result<T>
    where
        T: DeserializeOwned;
}

trait EmbeddingProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}
```

The v1 concrete implementation talks to `mistral.rs` using its OpenAI-compatible API.

## 3.3 Derived work must be cacheable

Every expensive operation must be keyed by enough information to know whether its result is still valid.

At minimum, cache keys should incorporate:

- input content hash;
- prompt/template version;
- model identifier;
- relevant model settings;
- operation/stage name.

Do not recompute unchanged work after a resume.

## 3.4 Never destroy a tab merely because processing reached a later stage

Closing a browser tab is an external side effect and must occur only after:

1. the tab was selected by the run;
2. processing completed successfully;
3. the close blacklist allows closing;
4. the final output artifact exists or the server has otherwise recorded successful completion;
5. the extension verifies the current tab still refers to the captured page;
6. the user requested automatic closing for this run.

If the tab has navigated to a different URL since capture, do not close it.

---

# 4. Browser extension responsibilities

Build the extension with **WXT**. Keep the extension source browser-independent and generate target-specific builds with WXT. Use WXT's unified `browser` API in application code rather than direct `chrome.*` or `browser.*` globals. WXT supports building the same extension codebase for Chrome, Firefox, Edge, Safari, and Chromium-based browsers, while allowing target-specific manifest/entrypoint configuration. citeturn688921search0turn688921search2turn688921search5

Do not assume all WebExtension APIs have identical behavior or availability across browsers. Keep unavoidable differences inside WXT configuration or very small browser-adapter modules. For example, browser sidebars/panels differ between Chrome and Firefox, so the v1 UI should use the popup/options pages and local dashboard rather than depend on a browser-specific sidebar API. citeturn688921search1

Use the WebExtensions tabs/scripting APIs through WXT's `browser` abstraction for tab enumeration, content extraction, and tab closing. The core workflow must not depend on Chrome-only APIs.

## 4.1 Extension UI

The toolbar popup should present:

### Scope

- **Everything** — all eligible tabs in all normal browser windows.
- **Selected windows** — show browser windows and allow window selection.
- **Selected tabs** — allow individual tab selection.

Optionally include **Current window** as a convenience even though it is not strictly required.

### Taxonomy

Show the currently configured taxonomy mode and allow the user to choose: 

- **Auto-discover** — no predefined taxonomy.
- **Predefined: preferred** — select one of the server's imported taxonomies.
- **Predefined: strict** — select one of the server's imported taxonomies.

The extension should fetch the available taxonomy list from the server and include the selected taxonomy ID in run creation. It must not implement taxonomy logic itself.

### Closing

- `Do not close tabs`
- `Close tabs after successful processing`

If closing is enabled, the close blacklist is still authoritative.

### Controls

- Start organizing.
- Show active run/progress.
- Resume/reconnect to an existing run if the server reports one.
- Open local dashboard.
- Open settings.

## 4.1.1 WXT requirements

Recommended extension layout:

```text
extension/
├── wxt.config.ts
├── entrypoints/
│   ├── background.ts
│   ├── popup/
│   ├── options/
│   └── content.ts
├── components/
├── lib/
│   ├── api-client.ts
│   ├── tab-capture.ts
│   ├── blacklist.ts
│   └── browser-adapter.ts
└── assets/
```

The core modules should use `browser.*` from `wxt/browser`. WXT documents this as the unified API over the runtime's browser globals. citeturn688921search0

Generate browser-specific builds with WXT, for example `wxt -b chrome`, `wxt -b firefox`, and `wxt -b edge`. Keep per-browser manifest permissions and compatibility workarounds in WXT configuration rather than duplicating the extension source. citeturn688921search2turn688921search8

For v1, Chrome/Chromium and Firefox are acceptance targets; Edge is an additional build target. WXT can build Safari targets, but Safari distribution currently requires wrapping the generated Web Extension with Apple's Safari tooling/Xcode, so do not make Safari packaging a v1 release gate. citeturn688921search11

## 4.2 Settings

The extension settings page should configure:

### Processing blacklist

Entries can be:

- exact URL;
- URL prefix/pattern;
- exact hostname;
- hostname suffix/domain, e.g. `example.com` matching subdomains according to explicit documented semantics.

Examples:

```text
chrome://*
localhost
bank.example.com
example.com/private/*
```

### Close blacklist

Same pattern types as above.

Important semantics:

- A processing-blacklisted tab is never sent for analysis.
- A processing-blacklisted tab must never be closed by this run.
- A close-blacklisted tab may be fully processed and written, but must remain open.
- Processing blacklist has precedence over everything else.
- Close blacklist is evaluated only when deciding the final close action.

Do not implement arbitrary regex as the only matching mechanism in v1. Use a small explicit pattern language that is easy to explain and test.

## 4.3 Capture behavior

When the user starts a run, the extension first creates a **snapshot of the selected scope**.

For every eligible tab, capture at least:

```json
{
  "browser": "chrome|firefox|edge|safari|other",
  "browser_tab_id": "opaque browser-provided identifier",
  "window_id": 456,
  "tab_index": 3,
  "url": "https://example.com/article",
  "title": "Article title",
  "pinned": false,
  "incognito": false,
  "captured_at": "..."
}
```

Then extract page content from the rendered page.

Preferred extraction order:

1. Mozilla Readability-style main-content extraction (bundle the required JS into the extension).
2. Fallback to cleaned `document.body.innerText`.
3. If extraction fails, send metadata with an explicit extraction failure status; the server may attempt an HTTP fetch fallback for publicly accessible pages if implemented.

Do not send the complete DOM to the server unless unavoidable. Prefer cleaned readable text.

## 4.4 Unsupported/special pages

Recognize that some URLs cannot be scripted, such as browser-internal pages and extension pages.

Examples that should normally be skipped with an explicit reason:

- browser-internal schemes such as `chrome://...`, `edge://...`, `about:...`, and analogous internal pages
- extension-internal schemes
- browser-store pages if scripting access is unavailable
- other browser-internal or privileged schemes

Do not classify these as failed AI processing. They are `unsupported`/`skipped`.

For PDFs displayed in the browser's viewer, v1 may mark them unsupported unless a reliable extraction path is implemented.

## 4.5 Sending data

Send captures incrementally rather than waiting for all 1,000 tabs to finish extraction.

Recommended flow:

```text
Create run
   |
   v
Capture scope snapshot
   |
   +--> tab capture 1 --> POST /runs/:id/items
   +--> tab capture 2 --> POST /runs/:id/items
   +--> ...
```

The server should assign an immutable run item ID.

The extension may batch a small number of captures per request, but the server must process each item independently so an interrupted batch can be retried safely.

---

# 5. Local application server API

Implement a versioned JSON HTTP API. Use a framework such as Axum.

Base path:

```text
/api/v1
```

## 5.1 Health/config endpoints

```text
GET /api/v1/health
GET /api/v1/config
```

`health` should indicate whether the application is available and whether the LLM/embedding provider is reachable.

## 5.2 Run lifecycle

```text
POST /api/v1/runs
GET  /api/v1/runs
GET  /api/v1/runs/:run_id
POST /api/v1/runs/:run_id/cancel
POST /api/v1/runs/:run_id/resume
GET  /api/v1/runs/:run_id/events
```

`POST /runs` request should contain:

```json
{
  "scope": {
    "kind": "all_tabs | window_ids | tab_ids",
    "window_ids": [],
    "tab_ids": []
  },
  "close_after_processing": true,
  "processing_blacklist": [],
  "close_blacklist": [],
  "taxonomy": {
    "mode": "auto | predefined_preferred | predefined_strict",
    "taxonomy_id": null
  }
}
```

The extension resolves the selected browser scope and sends the snapshot information. The server must persist that scope snapshot as part of the run.

Do not let later browser changes redefine what the original run meant to process.

## 5.3 Tab/capture endpoints

```text
POST /api/v1/runs/:run_id/items
GET  /api/v1/runs/:run_id/items
GET  /api/v1/runs/:run_id/items/:item_id
POST /api/v1/runs/:run_id/items/:item_id/retry
```

Each item should contain browser metadata and extracted page content.

## 5.4 Results/review endpoints

```text
GET /api/v1/runs/:run_id/summary
GET /api/v1/runs/:run_id/tags
GET /api/v1/runs/:run_id/notes
```

Include enough data for a simple dashboard/review page.

## 5.5 Taxonomy endpoints

Support predefined taxonomies as first-class server resources:

```text
GET    /api/v1/taxonomies
POST   /api/v1/taxonomies
GET    /api/v1/taxonomies/:taxonomy_id
PUT    /api/v1/taxonomies/:taxonomy_id
DELETE /api/v1/taxonomies/:taxonomy_id
POST   /api/v1/taxonomies/validate
```

The extension does not need to understand taxonomy internals. Taxonomy import, validation, storage, versioning, and use during a run belong to the app/server.

Provide a simple server/dashboard or CLI import path in v1, for example:

```text
tab-organizer taxonomy import ./taxonomy.yaml
tab-organizer taxonomy list
tab-organizer taxonomy validate ./taxonomy.yaml
```

The extension only lists and selects already-imported taxonomies.

---

# 6. Run state machine and resumability

This is a required feature, not an optimization.

## 6.1 Run-level states

```text
CREATED
CAPTURING
PROCESSING
FINALIZING
COMPLETED
CANCELLED
FAILED
```

A run can resume from an interrupted `CAPTURING`, `PROCESSING`, or `FINALIZING` state.

## 6.2 Item-level states

Recommended state machine:

```text
DISCOVERED
  |
  v
CAPTURED
  |
  v
EXTRACTED
  |
  v
SUMMARIZED
  |
  v
EMBEDDED
  |
  v
CLASSIFIED
  |
  v
WRITTEN
  |
  v
CLOSE_ELIGIBLE
  |
  v
CLOSED
```

Terminal non-success states:

```text
SKIPPED_BLACKLIST
SKIPPED_UNSUPPORTED
EXTRACTION_FAILED
ANALYSIS_FAILED
WRITE_FAILED
CLOSE_FAILED
```

Failures should be retryable without invalidating the whole run.

## 6.3 Idempotency

Every processing stage should be safe to execute more than once.

Examples:

- If summary already exists for `(content_hash, prompt_version, model_id)`, reuse it.
- If embedding already exists for `(analysis_input_hash, embedding_model)`, reuse it.
- If Markdown output already exists and the generated note content hash matches, do not rewrite it.
- If a close request was attempted and the tab is already gone, treat the result as effectively closed only after verifying that it disappeared.

## 6.4 Run recovery

On startup:

1. Load non-terminal runs.
2. Mark stale worker leases as expired.
3. Resume incomplete work.
4. Do not duplicate outputs or duplicate LLM requests when cache entries exist.

Use worker leases/heartbeats or an equivalent mechanism so the same item is not processed concurrently by multiple workers after a crash.

## 6.5 Retry policy

Transient failures:

- HTTP connection errors
- temporary model unavailable
- timeout
- rate/queue-style temporary errors

should be retried with exponential backoff.

Permanent failures should be recorded with a useful machine-readable reason.

Support:

```text
retry failed items
retry one item
resume run
cancel run
```

---

# 7. Storage model for v1

SQLite is the default local adapter, but must be hidden behind repository interfaces.

Suggested records/tables for the SQLite implementation:

```text
runs
run_items
tab_captures
document_content
analyses
embeddings
clusters
tags
tag_aliases
document_tags
notes
llm_cache
jobs_or_leases
settings
taxonomies
taxonomy_concepts
taxonomy_aliases
```

The exact SQL schema is an implementation detail.

The business layer should ask questions such as:

```text
get_run
save_run
list_pending_items
get_analysis_for_key
save_analysis
get_embedding_for_key
save_embedding
get_tags
save_tag
save_assignment
record_output
```

not:

```text
SELECT ...
INSERT INTO ...
```

## 7.1 Filesystem output is also separate from the database

Generated Markdown notes should be stored in a user-selected local directory.

The DB records the output location and content hash, but the DB is not the Markdown vault.

---

# 8. Document preprocessing

For each captured tab:

1. Normalize URL.
2. Record exact source URL as captured.
3. Compute canonical URL where practical.
4. Normalize extracted text.
5. Remove obvious repeated navigation/footer/cookie content if the extractor has not already done so.
6. Compute content hash.
7. Detect near-identical captures where practical.

Store:

- raw captured title;
- canonical title proposed by extractor/model if different;
- URL;
- canonical URL;
- extraction status;
- content hash;
- content length;
- capture timestamp.

Deduplication should occur before expensive LLM work whenever possible.

---

# 9. External predefined taxonomy

External predefined taxonomies are a supported v1 feature. The taxonomy should be treated as a reusable semantic vocabulary, not as a list of literal hashtag strings.

The run supports three modes:

```text
AUTO
    Discover a reusable vocabulary from the selected corpus.

PREDEFINED_PREFERRED
    Prefer and reuse concepts from the supplied taxonomy, but allow
    proposals for genuinely missing concepts when corpus evidence is strong.

PREDEFINED_STRICT
    Assign only concepts from the supplied taxonomy. A document may
    remain untagged when no supplied concept is adequate.
```

## 9.1 Input format

For v1 support local YAML and JSON files. Example:

```yaml
id: software-engineering
version: 1
name: Software Engineering
concepts:
  - id: rust-programming
    label: Rust programming
    description: Rust language, compiler, tooling, and ecosystem.
    aliases:
      - rust-lang
      - rust programming
    parent: programming

  - id: rust-chemistry
    label: Rust chemistry
    description: Rust formation, iron oxidation, and corrosion.
    aliases:
      - iron rust
      - rust corrosion
    parent: chemistry
```

Required concept fields:

- stable concept ID;
- canonical label;
- description.

Optional fields:

- aliases;
- parent;
- positive examples;
- negative examples;
- external IDs;
- arbitrary metadata.

The taxonomy file must be schema-validated and normalized before use.

## 9.2 Versioning and run reproducibility

Persist these values with every run that uses a taxonomy:

- taxonomy ID;
- taxonomy version;
- taxonomy content hash;
- taxonomy mode.

The run must continue using the recorded taxonomy snapshot even if the source file changes later.

## 9.3 Semantics

Keep these concepts distinct:

```text
concept ID
canonical label
alias
output tag representation
```

A concept such as `rust-programming` is semantic identity. Markdown may render it as `#rust-programming`, while a future PKMS adapter could render it differently.

## 9.4 Preferred-mode rules

In `PREDEFINED_PREFERRED` mode:

1. Prefer an existing predefined concept over inventing a synonymous concept.
2. Merge/map discovered concepts onto predefined concepts whenever they are semantically equivalent.
3. Permit a new concept only if the corpus contains recurring evidence and no predefined concept is semantically adequate.
4. New concepts are proposals until explicitly accepted.

In `PREDEFINED_STRICT` mode, the LLM is not allowed to invent a tag outside the supplied vocabulary.

## 9.5 Taxonomy quality reporting

At the end of a run report:

- documents tagged with predefined concepts;
- documents with no adequate concept;
- proposed new concepts;
- concept assignment counts;
- zero-use concepts;
- ambiguous/conflicting assignments.

# 10. LLM model strategy

## 10.1 Primary local model

Start with:

```text
Qwen3-14B
4-bit quantization
```

through native `mistral.rs`.

For this tab-organizer workload, default to **non-thinking mode** for routine per-document work.

Use thinking selectively for:

- difficult tag ambiguity;
- proposing/renaming a cluster concept;
- merging or splitting similar tags;
- resolving conflicts between plausible existing concepts;
- other low-confidence cases.

Do not make every one of 1,000 documents pay the cost of thinking.

## 10.2 Embeddings

Start with:

```text
Qwen3-Embedding-0.6B
```

served by `mistral.rs` through its embeddings endpoint.

If running text and embedding models simultaneously causes undesirable memory pressure, support separate model-server instances or a configurable load strategy.

## 10.3 Model server assumptions

The application should see an OpenAI-compatible provider interface. Configuration should include:

```text
LLM_BASE_URL
LLM_MODEL
EMBEDDING_BASE_URL
EMBEDDING_MODEL
```

Do not hardcode port numbers or model names into business logic.

---

# 11. Tagging algorithm

The tagging system is the most important part of the prototype.

The goal is NOT:

> Generate clever keywords independently for every document.

The goal is:

> Build a small reusable vocabulary for the whole tab collection, then classify documents against that vocabulary.

The algorithm should optimize for:

- semantic correctness;
- reuse;
- few singletons;
- clear disambiguation;
- small number of tags per document;
- stability across runs.

## 11.1 Stage A — select taxonomy mode

Resolve and validate the run's taxonomy mode before expensive LLM work. Load the predefined taxonomy snapshot, if any, and record its content hash.

```text
AUTO
    corpus discovery -> candidate taxonomy -> classification

PREDEFINED_PREFERRED
    predefined concepts + corpus evidence -> optional additions -> classification

PREDEFINED_STRICT
    predefined concepts -> candidate retrieval -> classification
```

## 11.2 Stage B — document analysis

For every unique extracted document, call Qwen3-14B with non-thinking mode and request structured JSON:

```json
{
  "summary": "1-3 concise paragraphs",
  "topics": [
    "candidate concept 1",
    "candidate concept 2",
    "candidate concept 3"
  ],
  "entities": [
    "...
  ],
  "document_type": "technical article"
}
```

Topics should describe durable concepts, not merely salient nouns.

Bad:

```text
rust
cargo
compiler
```

Better:

```text
Rust programming
memory safety
systems programming
```

The prompt should explicitly distinguish ambiguous concepts where context matters.

## 11.3 Stage C — embeddings

Create an embedding input such as:

```text
Title: <title>
Summary: <summary>
Topics: <topic 1>; <topic 2>; ...
```

Embed each document.

Also embed topic candidates when useful for candidate-topic clustering.

Store embeddings keyed by the exact input hash and embedding model.

## 11.4 Stage D — corpus-level concept discovery

Use the embeddings to identify recurring semantic groups.

For a ~1,000-document corpus, an O(N^2) similarity approach is acceptable if it keeps the implementation simple.

Start with one of:

- hierarchical agglomerative clustering using cosine similarity; or
- HDBSCAN / an available Rust implementation that behaves well for this corpus.

The implementation may choose the simpler robust option after benchmarking.

Important behavior:

- allow outliers;
- do not force every document into a cluster;
- preserve cluster membership scores/distances;
- expose cluster representatives to the LLM.

## 11.5 Stage E — propose or reconcile canonical tags

For every meaningful cluster, give Qwen3-14B representative summaries/topics and ask it to propose:

```json
{
  "canonical_tag": "rust-programming",
  "display_name": "Rust programming",
  "description": "The Rust programming language and ecosystem.",
  "aliases": ["rust-lang", "Rust programming"],
  "parent": "programming",
  "why_this_cluster": "..."
}
```

The model should prefer reusable concepts over page-specific wording.

## 11.6 Stage F — canonicalize and merge tags

Create a candidate tag registry.

Compare candidate tags semantically and lexically.

Ask the LLM in **thinking mode only for ambiguous merges**.

Examples:

```text
rust-lang           -> rust-programming
rust programming    -> rust-programming
systems programming -> systems-programming
```

but:

```text
rust-programming   != rust-chemistry
```

Tag identity should depend on semantic definitions, not merely aliases/words.

## 11.7 Stage G — enforce support/reuse policy

Use corpus-level support counts.

Suggested starting policy:

```text
minimum support for new tag: 3 documents
preferred support:            5+
max tags/document:            5
```

Singletons should be rejected by default.

A singleton may be retained only if the system explicitly marks it as a high-value exceptional concept. For v1, it is acceptable to reject all singleton-generated tags and leave the page with fewer tags.

Never generate a generic catch-all tag such as:

```text
miscellaneous
other
random
web
interesting
```

merely to avoid an untagged document.

## 11.8 Stage H — final document classification

For every document:

1. Embed the document representation.
2. Retrieve the nearest candidate canonical tags.
3. Provide those candidate tags plus their definitions/examples to Qwen3-14B.
4. Ask it to choose 0–5 tags.
5. Default to non-thinking mode.
6. Escalate to thinking mode when:
   - top candidates are close;
   - confidence is low;
   - candidate concepts are semantically conflicting;
   - a new tag is being proposed.

Do NOT show the entire tag vocabulary to the model if retrieval can reduce it to a small candidate set.

When a predefined taxonomy is active, candidate retrieval must come from that taxonomy first. In strict mode, the model may only select from retrieved predefined concepts.

## 11.9 Stage I — global cleanup

After classification, calculate:

- tag frequency;
- number of singleton tags;
- mean tags/document;
- documents with no tags;
- candidate duplicate tags;
- tags whose assigned documents are semantically incoherent.

Then perform a cleanup pass.

For example:

```text
#rust
#rust-lang
#rust-programming
```

should generally collapse to one canonical concept.

If cleanup merges or removes a tag, re-run classification only for affected documents.

## 11.10 Tag scoring concept

A conceptual score should incorporate:

```text
semantic similarity
+ corpus support
+ compatibility with existing taxonomy
- singleton penalty
- redundancy with already-selected tags
```

The exact formula may be heuristic in v1.

The most important behavior is that a tag needs to be both **semantically relevant** and **worth maintaining as a reusable organizational concept**.

---

# 12. Ambiguity resolution

Explicitly test ambiguous terms.

Required test case:

```text
Rust programming article
vs.
iron rust / corrosion article
```

The classifier must produce distinct concepts.

Other useful test cases:

```text
python the language
vs.
python snake

apple software company
vs.
apple fruit

mercury planet/science
vs.
mercury element

shell programming shell
vs.
shell mollusk
```

Do not rely on token matching for these decisions.

Tag definitions should contain positive and negative examples when useful:

```yaml
rust-programming:
  description: Rust programming language and ecosystem
  positive_examples:
    - ownership and borrowing
    - Cargo
  negative_examples:
    - iron corrosion

rust-chemistry:
  description: rust formation and corrosion of iron/steel
  positive_examples:
    - oxidation of iron
    - corrosion
  negative_examples:
    - Cargo
    - borrowing
```

---

# 13. Markdown output

Each successfully processed tab should produce a Markdown note.

Recommended structure:

```markdown
---
title: "Understanding Rust's Ownership Model"
source_url: "https://example.com/rust"
canonical_url: "https://example.com/rust"
captured_at: "2026-10-07T..."
tags:
  - programming/rust
  - systems/memory-safety
---

# Understanding Rust's Ownership Model

Rust's ownership and borrowing system provides memory safety without
requiring a garbage collector. The article explains ownership,
references, borrowing, lifetimes, and compiler enforcement.

## Source

https://example.com/rust
```

For v1, keep the summary concise and useful rather than reproducing the article.

Do not include model reasoning in the note.

Use deterministic, filesystem-safe filenames. Example:

```text
understanding-rusts-ownership-model--<short-id>.md
```

Do not derive identity solely from the filename.

---

# 14. Browser compatibility and extension tests

The extension must be exercised from the same WXT source tree against:

- Chrome/Chromium;
- Firefox;
- Edge build.

Safari should have a documented build path, but Safari packaging is not a v1 release gate because WXT's current Safari distribution flow requires an Apple/Xcode wrapper. citeturn688921search11

Tests must cover:

- enumerate all normal windows/tabs;
- select one or more windows;
- select individual tabs;
- capture URL/title/index/window ID/browser identity;
- list predefined taxonomies and select a taxonomy/mode for a run;
- extract readable content;
- send captures to the localhost server;
- receive progress updates;
- resume after extension restart;
- close processed tabs;
- refuse to close a navigated tab;
- processing blacklist matching;
- close blacklist matching;
- pinned-tab protection;
- browser-internal page handling;
- browser-specific API differences isolated behind adapters.

The extension's core workflow must contain no browser-specific calls. Any target-specific branch must be easy to find and separately tested.

# 15. Closing tabs safely

This feature must be conservative.

## 15.1 Close eligibility

A tab is close-eligible only when all of the following are true:

```text
run requested close_after_processing
AND
item processed successfully
AND
Markdown/output artifact successfully recorded
AND
tab does not match close blacklist
AND
current browser tab still corresponds to captured URL
AND
tab is not pinned unless the user explicitly allows pinned-tab closing
```

For v1, **do not close pinned tabs by default**.

## 15.2 Close handshake

The server must not itself attempt to close Chrome tabs.

Instead:

```text
Server:
  item 123 is CLOSE_ELIGIBLE
        |
        v
Extension:
  query current tab 123
        |
        +-- URL unchanged --> close
        |
        +-- URL changed ----> leave open + report changed
        |
        +-- tab gone --------> mark already closed
```

The extension should report the final close result back to the server.

## 15.3 Close blacklist

A matching close-blacklist rule overrides close eligibility.

Report a clear reason:

```text
processed successfully, not closed: close-blacklist
```

---

# 16. Blacklist semantics

Use a shared matcher abstraction for both blacklists.

Pattern types:

```text
ExactUrl
UrlPrefix
ExactHost
HostSuffix
```

Normalize hostnames to lowercase and handle leading `www.` according to clearly documented semantics.

For v1, avoid surprising substring matching.

Example:

```text
example.com
```

should match `www.example.com` and `docs.example.com` if configured as a host-suffix rule, but should not match `notexample.com`.

The run records the blacklist configuration used at run creation so that resuming later does not silently reinterpret the original scope.

---

# 17. Concurrency and worker model

The application server should use a bounded async worker pool.

Recommended initial behavior:

- low concurrency for Qwen3-14B generation;
- batching for embeddings when supported;
- configurable maximum in-flight LLM requests;
- backpressure when the model server is saturated.

Do not try to maximize parallelism on the M1 Pro blindly. The goal is stable throughput and predictable memory use.

Suggested initial configuration:

```text
LLM concurrency:       1
Embedding batch size:  16–32
CPU worker concurrency: modest, configurable
```

Benchmark before raising LLM concurrency.

---

# 18. LLM prompt design

All prompts must have explicit versions, e.g.:

```text
summary-v1
analysis-v1
tag-discovery-v1
tag-merge-v1
classification-v1
```

The prompt version must participate in cache keys.

Require structured JSON for machine-consumed outputs.

Do not accept malformed free-form model output as successful structured analysis.

Retry malformed outputs once with a repair prompt; after that mark the item failed.

For routine processing, use `/no_think`.

Use `/think` selectively and record whether thinking mode was used in provenance/debug metadata.

---

# 19. Quality controls

The prototype should produce metrics at the run level.

At minimum:

```text
Total selected tabs
Processed
Skipped by process blacklist
Skipped unsupported
Extraction failures
LLM failures
Written notes
Close-eligible
Closed
Close-blacklisted
Changed-before-close

Distinct tags
Median tag frequency
Singleton tag count
Average tags/document
Documents without tags
```

Also record timing:

```text
capture time
extraction time
LLM time
embedding time
clustering time
writing time
```

This will make model/prompt experiments much easier.

---

# 20. Review/dashboard requirements

A full desktop UI is not required.

Provide a minimal local web dashboard from the application server or a static local UI.

It should show:

- current run;
- progress by stage;
- failed items;
- skipped items and reasons;
- resulting tag vocabulary;
- tag frequencies;
- documents per tag;
- proposed tag merges;
- whether tabs will be closed.

Before applying final tag changes/output, the user should be able to inspect the tag vocabulary.

A small approval step for the tag vocabulary is highly desirable.

---

# 21. Distribution strategy

## 19.1 Prototype/dev distribution

Use Docker Compose for the application server.

Provide:

```text
docker compose up -d
```

and a documented native `mistral.rs` setup.

The repository should include:

```text
Dockerfile
compose.yaml
.env.example
README.md
scripts/
```

## 19.2 One-command developer setup

Provide a script that checks:

- Docker available;
- `mistralrs` available;
- required model access/configuration;
- ports available;
- writable data directory;
- server health.

The script should not silently expose services to the LAN.

## 19.3 Future average-user packaging

Do NOT build the polished installer in the first implementation milestone.

However, design for a future macOS desktop launcher that can:

1. install/upgrade the application;
2. start/stop the server container automatically;
3. install/configure `mistral.rs` natively;
4. download models;
5. install/register the browser extension or provide the appropriate installation flow;
6. open a local setup UI;
7. provide repair/reset diagnostics.

The eventual UX should hide Docker from average users.

Power users should still be able to use raw Docker Compose, CLI configuration, or their own LLM server.

---

# 22. Security and privacy requirements

All v1 services are local-only by default.

Application server:

```text
bind 127.0.0.1
```

or an equally restricted host-only configuration.

Model server:

- do not expose publicly;
- configure access according to the Docker-to-host topology;
- document any required non-loopback bind explicitly.

Do not transmit page content to a cloud provider in the default configuration.

The application should display an explicit warning if the configured LLM endpoint is not local.

Use an application authentication token for extension-to-server requests even though the server is localhost-only. Do not rely solely on the fact that a port is local.

The extension should store its token using browser extension storage and should never expose it to page scripts.

Validate request origins/senders where applicable.

---

# 23. Error handling philosophy

Errors should be categorized.

```text
recoverable
retryable
skippable
permanent
user-action-required
```

Examples:

```text
chrome:// page
  -> skippable / unsupported

page extraction timed out
  -> retryable

LLM server unavailable
  -> run paused/retryable

malformed LLM JSON after retry
  -> permanent for this item

output directory inaccessible
  -> user-action-required

URL changed before close
  -> non-fatal / do-not-close
```

A single bad tab must never cause the whole 1,000-tab run to fail.

---

# 24. Testing strategy

Tests are required throughout the prototype.

## 22.1 Unit tests

Test:

- URL canonicalization;
- blacklist matching;
- close-blacklist semantics;
- state transitions;
- idempotency keys;
- content hashing;
- filename sanitization;
- tag support counts;
- singleton pruning;
- tag deduplication;
- candidate selection;
- confidence thresholds;
- organization rules.

## 22.2 Tagging fixtures

Create a small corpus of hand-labeled synthetic/realistic text fixtures that includes:

```text
Rust programming
iron rust / corrosion
Python language
Python snake
Apple company
apple fruit
Linux kernel
Linux desktop
machine learning
traditional statistics
```

The system should demonstrably distinguish ambiguous concepts.

## 22.3 Resume/crash tests

Simulate failures:

- process killed during summarization;
- killed after embedding but before classification;
- killed during Markdown writing;
- killed after output but before close;
- model server unavailable and then restored;
- extension disconnected for several minutes.

Resume must not duplicate notes or lose successful work.

## 22.4 Browser tests

Test:

- all windows;
- selected windows;
- selected tabs;
- tabs in multiple windows;
- pinned tabs;
- blacklisted tabs;
- close-blacklisted tabs;
- unsupported pages;
- tab navigation during processing;
- tab closed by user during processing.

---

# 25. Acceptance criteria for v1

A prototype is considered functionally complete when all of the following are true.

### Browser

- [ ] Extension can enumerate all eligible tabs.
- [ ] Extension can organize everything.
- [ ] Extension can organize selected windows.
- [ ] Extension can organize selected tabs.
- [ ] Extension can capture rendered page text for normal web pages.
- [ ] Unsupported pages are reported rather than crashing the run.
- [ ] Extension can request close-after-processing.
- [ ] Extension can keep all tabs open.
- [ ] Extension enforces process blacklist.
- [ ] Extension enforces close blacklist.

### Server

- [ ] Server runs locally.
- [ ] Extension communicates with server.
- [ ] Server persists run state.
- [ ] Server resumes incomplete runs after restart.
- [ ] Individual failed items can be retried.
- [ ] Work is idempotent.
- [ ] Run progress is observable.

### LLM

- [ ] Server communicates with native `mistral.rs`.
- [ ] Qwen3-14B structured summaries work.
- [ ] Embeddings work.
- [ ] Non-thinking mode is the default for routine analysis.
- [ ] Thinking is used selectively for hard cases.

### Tagging

- [ ] Tags are discovered across the whole corpus.
- [ ] Existing/canonical tags are preferred over synonyms.
- [ ] Singleton tags are rare or absent.
- [ ] Duplicate/synonymous tags are merged.
- [ ] Ambiguous concepts are separated.
- [ ] Documents can legitimately receive zero tags.
- [ ] Documents normally receive a small number of tags (target <= 5).

### Browser independence

- [ ] Extension source is implemented with WXT.
- [ ] One WXT source tree builds Chrome/Chromium and Firefox without source forks.
- [ ] Edge builds from the same source tree.
- [ ] Extension can list and select server-provided predefined taxonomies.
- [ ] Core extension code uses WXT's unified `browser` API rather than direct `chrome.*` globals.
- [ ] Browser-specific manifest/API differences are isolated in WXT configuration or a small adapter.
- [ ] Safari compatibility/build path is documented; Safari packaging is outside the v1 release gate.

### Predefined taxonomy

- [ ] User can import a local YAML or JSON taxonomy.
- [ ] Taxonomy is schema-validated and content-hashed.
- [ ] Taxonomy ID/version/hash/mode are recorded with each run.
- [ ] Preferred mode maximizes reuse of predefined concepts while allowing carefully justified new concepts.
- [ ] Strict mode never produces tags outside the predefined vocabulary.
- [ ] Taxonomy changes do not retroactively change completed run results.

### Output

- [ ] Every successfully processed tab produces a Markdown note.
- [ ] Notes contain title, URL, summary, and hashtags/tags.
- [ ] Notes are deterministic enough to avoid duplicate creation on resume.

### Safety

- [ ] Tabs are never closed unless explicitly requested.
- [ ] Tabs are never closed when they match the close blacklist.
- [ ] Tabs changed since capture are left open.
- [ ] Pinned tabs are left open by default.
- [ ] Application and model endpoints are local-only by default.

---

# 26. Recommended implementation milestones

## Milestone 1 — skeleton

Deliver:

- Rust workspace/project;
- application server;
- health endpoint;
- Docker image + Compose;
- configuration system;
- repository interfaces;
- SQLite adapter;
- native mistral.rs connectivity check.

Success criterion: server starts in Docker and can make one structured request to local `mistral.rs`.

## Milestone 2 — extension capture

Deliver:

- Manifest V3 extension;
- enumerate tabs;
- scope selection UI;
- rendered text extraction;
- send captures to server;
- capture errors/unsupported states.

Success criterion: capture ~100 representative tabs reliably.

## Milestone 3 — resumable processing

Deliver:

- run/item states;
- persistence;
- checkpoints;
- retries;
- startup recovery;
- job progress.

Success criterion: kill/restart the server several times during processing and resume without duplicate work.

## Milestone 4 — summaries + embeddings

Deliver:

- Qwen3-14B structured analysis;
- non-thinking mode default;
- selective thinking path;
- Qwen3 embeddings;
- cached analysis/embeddings.

Success criterion: process a representative 100-tab corpus with deterministic cached reruns.

## Milestone 5 — tagging engine

Deliver:

- corpus embedding/clustering;
- candidate topic extraction;
- canonical tag proposals;
- tag merging;
- support-count policy;
- final classification;
- ambiguity fixtures.

Success criterion: the system creates a compact, reusable tag vocabulary and correctly separates known ambiguous pairs.

## Milestone 6 — Markdown writer

Deliver:

- deterministic note naming;
- YAML frontmatter;
- summary/body generation;
- output tracking;
- idempotent rewrites.

Success criterion: a run creates a usable set of Markdown notes that can be dropped into Obsidian.

## Milestone 7 — safe close behavior

Deliver:

- close-after-processing option;
- close blacklist;
- extension/server close handshake;
- navigation-race protection;
- pinned-tab protection.

Success criterion: intentionally mutate tabs during a run and verify that only the correct tabs close.

## Milestone 8 — review/dashboard

Deliver:

- run status page;
- tag vocabulary review;
- item failure/retry view;
- metrics.

Success criterion: user can inspect the run and understand what happened without reading logs.

## Milestone 9 — real-world 1,000-tab trial

Run against the user's actual tab collection.

Do NOT immediately close anything.

First run should generate notes and a report only.

After verifying quality, rerun a small subset with close-after-processing enabled.

Only after the close path has been proven should the full tab-bankruptcy run use automatic closing.

---

# 27. Agent coding instructions

The implementing agent should follow these rules:

1. Build a working vertical slice early; do not spend the first phase on generalized abstractions.
2. Keep persistence behind interfaces even though SQLite is the only v1 implementation.
3. Keep `mistral.rs` behind an LLM/embedding provider interface.
4. Favor explicit state machines and durable checkpoints over clever concurrency.
5. Prefer deterministic behavior and cached results.
6. Keep destructive browser actions separate from analysis.
7. Never auto-close on an uncertain or failed state.
8. Produce structured JSON from the LLM whenever the output is consumed by code.
9. Keep prompts versioned.
10. Make every expensive stage independently retryable.
11. Add tests for every browser-close safety rule before enabling automatic closing.
12. Do not implement generalized PKMS/domain abstractions yet; Markdown output is sufficient for v1.
13. Do not require cloud services.
14. Do not expose local services to the LAN by default.
15. Prefer simple, observable implementations over highly optimized ones for the first working prototype.

---

# 28. Open implementation decisions for the agent

The agent may choose among reasonable options without asking the user when the decision does not materially change behavior.

Examples:

- Axum vs another mature Rust HTTP framework.
- SQLite crate (`sqlx`, `rusqlite`, etc.) behind the persistence boundary.
- exact clustering implementation, provided it is Rust-native and handles ~1,000 items well;
- exact Markdown filename slugging algorithm;
- extension UI framework, provided the final extension remains small and maintainable.

The agent should ask the user only when a decision changes a user-visible product behavior or creates a significant security/privacy tradeoff.

---

# 29. Important design note on Docker + native mistral.rs

The proposed prototype deployment is sensible, with one caveat: Docker Desktop on macOS runs containers in a lightweight Linux VM, so the container does not simply share the host's loopback namespace in the default setup.

Use Docker Desktop's documented host access mechanism (`host.docker.internal`) or explicitly enabled host networking for the application container. Keep this topology configurable.

For the prototype, validate these exact scenarios early:

```text
browser extension
    -> localhost:APP_PORT
    -> Docker container
    -> host mistral.rs
```

and:

```text
Docker restart
    -> model remains available
    -> application reconnects
```

Do not assume `127.0.0.1` inside the container means the macOS host.

If the required host-service exposure becomes awkward or insecure, introduce a tiny native host-side bridge later rather than contaminating the application layer with Docker-specific networking assumptions.

---

# 30. First implementation task

Start by building the smallest vertical slice:

```text
Chrome extension
  -> capture one selected tab
  -> POST tab data
  -> Rust server persists it
  -> Rust server asks mistral.rs for a structured summary
  -> Rust server writes one Markdown note
  -> API reports success
```

Then immediately make that slice resumable/idempotent before scaling to 1,000 tabs.

Once the vertical slice works, add embeddings/clustering/tag vocabulary discovery, then closing.

The prototype should be useful long before the final architecture is complete.

---

## Reference material consulted while creating this plan

- Chrome for Developers: `chrome.tabs`, `chrome.scripting`, Manifest V3 permissions, and Native Messaging documentation.
- Docker Docs: Docker Desktop networking, `host.docker.internal`, localhost port publishing, and host networking on modern Docker Desktop.
- mistral.rs documentation: local API serving, OpenAI-compatible endpoints, structured output, embeddings, and Qwen3 / Qwen3-Embedding support.

# 31. Relevant implementation references

WXT:
- Extension APIs: https://wxt.dev/guide/essentials/extension-apis
- Targeting different browsers: https://wxt.dev/guide/essentials/target-different-browsers.html
- Manifest configuration: https://wxt.dev/guide/essentials/config/manifest.html
- Publishing: https://wxt.dev/guide/essentials/publishing

Cross-browser WebExtensions:
- MDN Chrome incompatibilities: https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Chrome_incompatibilities
- MDN WebExtensions API reference: https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API

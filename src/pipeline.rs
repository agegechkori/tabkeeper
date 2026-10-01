use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Context, Result};
use futures_util::future::ready;
use futures_util::stream::{self, StreamExt};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::config::{Config, TagConfig};
use crate::db::{Db, EmbeddingKind, PageResult, PendingPage, page_embedding_text, tag_embedding_text};
use crate::embed::{BATCH_SIZE, Embedder, TagIndex};
use crate::extract::{extract, html_title, truncate_chars};
use crate::fetch::Fetcher;
use crate::filter::Filter;
use crate::llm::{LangMode, Llm, PageRejected, PageSummary, SummaryRequest};
use crate::progress::Progress;
use crate::tags::{CountedTag, count_tree, format_vocabulary, is_offerable, most_used, normalize_name};

/// Reserved tags for the stub notes of pages without a summary.
pub const UNREACHABLE_TAG: &str = "status/unreachable";
pub const FAILED_TAG: &str = "status/failed";

enum Stub {
    /// The page couldn't be loaded (404, dead domain, timeout, blocked).
    Unreachable,
    /// The page loaded but couldn't be summarized (PDF, invalid model reply).
    Failed,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub done: usize,
    pub failed: usize,
    pub unreachable: usize,
    /// Pending pages the filter excludes; they stay pending.
    pub filtered: usize,
    /// Pages whose text was cut to fit `llm.max_input_chars`.
    pub pages_cut: usize,
    /// Pages that were sent to the model (done, or failed at the model).
    pub model_pages: usize,
    /// Unreachable and failed pages of this run by cause, e.g. `not_found`.
    pub causes: std::collections::BTreeMap<String, usize>,
}

/// Limits on what a run may spend; it stops starting pages once one is reached.
#[derive(Debug, Default, Clone, Copy)]
pub struct Budget {
    /// Input plus output tokens, summaries and embeddings together.
    pub max_tokens: Option<u64>,
    /// US dollars, using the prices in the config.
    pub max_cost: Option<f64>,
}

/// Why a run ended before all pending pages were processed.
#[derive(Debug)]
pub enum Stop {
    /// A problem that would affect every page; the run should exit with an error.
    Error(anyhow::Error),
    /// A --max-tokens or --max-cost limit was reached.
    Budget(String),
}

/// What a run did. Returned even when the run stopped early, so it can be
/// reported.
#[derive(Debug, Default)]
pub struct RunSummary {
    pub stats: Stats,
    /// Time spent fetching pages, added up over all pages.
    pub fetch_secs: f64,
    /// Time spent waiting for summaries, added up over all pages.
    pub model_secs: f64,
    pub stop: Option<Stop>,
}

/// What happened to one page.
enum Outcome {
    Done(String),
    /// Saved as a stub note: the cause, and what happened.
    Unreachable(&'static str, String),
    /// Saved as a stub note and recorded as failed; the page is skipped until
    /// `--retry-failed`.
    Failed(&'static str, String),
    /// Every page would fail (e.g. the LLM server is down): stop the run. The
    /// page stays pending.
    Fatal(anyhow::Error),
    /// Saved as failed, and the run should stop: too many pages in a row
    /// failed at the model.
    FailedAndStop(&'static str, String, anyhow::Error),
}

/// Embeddings for a run: the client plus every tag's vector in memory.
pub struct Embeddings<'a, E: Embedder> {
    embedder: &'a E,
    index: TagIndex,
    page_chars: usize,
    /// Embedding requests that failed during the run. The next run fills in
    /// any vectors that are still missing.
    pub errors: usize,
    /// (tag id, text) of new tags whose vectors couldn't be stored yet. They
    /// are retried with the next page, so later pages can still be shown them.
    unindexed_tags: Vec<(i64, String)>,
}

impl<'a, E: Embedder> Embeddings<'a, E> {
    /// Embeds the tags and pages that have no vector for this model yet (for
    /// example after switching models, or after an interrupted run) and
    /// loads all tag vectors. Fails if the embeddings server can't be used.
    pub async fn prepare(db: &Db, embedder: &'a E, page_chars: usize) -> Result<Self> {
        let model = embedder.model();
        // Checks the server and model even when there is nothing to backfill.
        embedder.embed(&["tabkeeper".to_string()]).await?;
        for (kind, missing) in [
            (EmbeddingKind::Tag, db.tags_missing_embedding(model)?),
            (EmbeddingKind::Page, db.pages_missing_embedding(model)?),
        ] {
            for chunk in missing.chunks(BATCH_SIZE) {
                let texts: Vec<String> = chunk.iter().map(|(_, text)| text.clone()).collect();
                let vectors = embedder.embed(&texts).await?;
                for ((id, _), vector) in chunk.iter().zip(vectors) {
                    db.put_embedding(kind, *id, model, &vector)?;
                }
            }
        }
        let index = TagIndex::new(db.embeddings(EmbeddingKind::Tag, model)?);
        Ok(Self {
            embedder,
            index,
            page_chars,
            errors: 0,
            unindexed_tags: Vec::new(),
        })
    }

    /// Records a failed embedding request. Only the first one is printed,
    /// so a flaky embeddings server doesn't flood the output.
    fn note_error(&mut self, progress: &Progress, what: &str, consequence: &str, err: &anyhow::Error) {
        if self.errors == 0 {
            progress.line(&format!(
                "Warning: {what} failed ({err:#}). Continuing: {consequence}. \
                 Further embedding errors are counted, not shown."
            ));
        }
        self.errors += 1;
    }
}

/// Limits how many pages are fetched from one domain at the same time.
struct DomainLimiter {
    per_domain: usize,
    semaphores: RefCell<HashMap<String, Arc<Semaphore>>>,
}

impl DomainLimiter {
    async fn acquire(&self, url: &str) -> OwnedSemaphorePermit {
        let host = url::Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(str::to_string))
            .unwrap_or_default();
        let semaphore = self
            .semaphores
            .borrow_mut()
            .entry(host)
            .or_insert_with(|| Arc::new(Semaphore::new(self.per_domain)))
            .clone();
        semaphore
            .acquire_owned()
            .await
            .expect("semaphores are never closed")
    }
}

/// What to process and how, for one run.
pub struct RunOptions<'a> {
    pub config: &'a Config,
    pub lang: &'a LangMode,
    pub filter: &'a Filter,
    /// Process at most this many pages.
    pub limit: Option<usize>,
    pub budget: Budget,
    /// Print progress to stderr, keeping stdout for a JSON report.
    pub progress_to_stderr: bool,
    /// Tokens and cost spent before the run by something it doesn't track,
    /// e.g. an embeddings setup that failed partway; counted toward the budget.
    pub spent_elsewhere: (u64, f64),
}

#[cfg(test)]
impl RunSummary {
    /// The counts, or the error if the run stopped on one (a budget stop isn't an error).
    pub fn into_result(self) -> Result<Stats> {
        match self.stop {
            Some(Stop::Error(err)) => Err(err),
            _ => Ok(self.stats),
        }
    }
}

/// Processes pending pages, several at a time. Each page is saved as soon as
/// it is done, so an interrupted run continues where it stopped. Without
/// embeddings, the model is shown the most used tags instead of the most
/// relevant ones.
pub async fn process_pending<F: Fetcher, L: Llm, E: Embedder>(
    db: &mut Db,
    fetcher: &F,
    llm: &L,
    embeddings: Option<&mut Embeddings<'_, E>>,
    options: &RunOptions<'_>,
) -> Result<RunSummary> {
    let RunOptions {
        config,
        lang,
        filter,
        limit,
        budget,
        progress_to_stderr,
        spent_elsewhere,
    } = *options;
    let mut stats = Stats::default();
    let mut pending = db.pending_pages()?;
    pending.retain(|page| {
        let allowed = filter.allows(&page.url);
        stats.filtered += usize::from(!allowed);
        allowed
    });
    if let Some(n) = limit {
        pending.truncate(n);
    }

    let progress = Progress::new(pending.len(), progress_to_stderr);
    let run = Run {
        db: RefCell::new(db),
        fetcher,
        llm,
        embeddings: RefCell::new(embeddings),
        config,
        lang,
        domains: DomainLimiter {
            per_domain: config.run.per_domain,
            semaphores: RefCell::default(),
        },
        progress: &progress,
        model_failures: Cell::new(0),
        fetch_secs: Cell::new(0.0),
        model_secs: Cell::new(0.0),
        pages_cut: Cell::new(0),
        model_pages: Cell::new(0),
        spent_before: Cell::new((0, 0.0)),
        spent_elsewhere,
    };
    run.spent_before.set(run.spent());
    let stats = RefCell::new(stats);
    let fatal: RefCell<Option<(String, anyhow::Error)>> = RefCell::new(None);
    let budget_stop: RefCell<Option<String>> = RefCell::new(None);
    let in_flight = Cell::new(0usize);
    let stopped = Cell::new(false);
    let run = &run;

    // The first pages go one at a time, so the tags they create are there for
    // the pages after them; the rest run concurrently.
    let first = pending.len().min(config.run.sequential_start);
    let (sequential, concurrent) = pending.split_at(first);
    for (pages, concurrency) in [(sequential, 1), (concurrent, config.run.concurrency)] {
        stream::iter(pages)
            .take_while(|_| {
                // Checked before each page starts, so a budget isn't overshot by
                // everything that was already started.
                if !stopped.get() && budget_stop.borrow().is_none() {
                    let finished = {
                        let s = stats.borrow();
                        s.done + s.unreachable + s.failed
                    };
                    if let Some(reason) = run.over_budget(&budget, in_flight.get(), finished) {
                        progress.line(&format!("Stopping: {reason}. Pages already started will finish."));
                        stopped.set(true);
                        *budget_stop.borrow_mut() = Some(reason);
                    }
                }
                ready(!stopped.get())
            })
            .map(|page| {
                in_flight.set(in_flight.get() + 1);
                async move { (page, run.process(page).await) }
            })
            .buffer_unordered(concurrency)
            .for_each(|(page, outcome)| {
                in_flight.set(in_flight.get() - 1);
                let mut stats = stats.borrow_mut();
                let line = match outcome {
                    Outcome::Done(title) => {
                        stats.done += 1;
                        format!("done         {} -> {title}", page.url)
                    }
                    Outcome::Unreachable(kind, why) => {
                        stats.unreachable += 1;
                        *stats.causes.entry(kind.to_string()).or_default() += 1;
                        format!("unreachable  {}: {why}", page.url)
                    }
                    Outcome::Failed(kind, why) => {
                        stats.failed += 1;
                        *stats.causes.entry(kind.to_string()).or_default() += 1;
                        format!("failed       {}: {why}", page.url)
                    }
                    Outcome::FailedAndStop(kind, why, stop) => {
                        stats.failed += 1;
                        *stats.causes.entry(kind.to_string()).or_default() += 1;
                        stopped.set(true);
                        fatal.borrow_mut().get_or_insert((page.url.clone(), stop));
                        format!("failed       {}: {why}", page.url)
                    }
                    Outcome::Fatal(err) => {
                        stopped.set(true);
                        fatal.borrow_mut().get_or_insert((page.url.clone(), err));
                        return ready(());
                    }
                };
                progress.line(&line);
                progress.advance(format!(
                    "{} done, {} unreachable, {} failed",
                    stats.done, stats.unreachable, stats.failed
                ));
                ready(())
            })
            .await;
    }
    progress.finish();

    let mut stats = stats.into_inner();
    stats.pages_cut = run.pages_cut.get();
    stats.model_pages = run.model_pages.get();
    let stop = match (fatal.into_inner(), budget_stop.into_inner()) {
        (Some((url, err)), _) => Some(Stop::Error(err.context(format!(
            "stopped at {url} ({} done, {} unreachable, {} failed so far; run again to continue)",
            stats.done, stats.unreachable, stats.failed
        )))),
        (None, Some(reason)) => Some(Stop::Budget(reason)),
        (None, None) => None,
    };
    Ok(RunSummary {
        stats,
        fetch_secs: run.fetch_secs.get(),
        model_secs: run.model_secs.get(),
        stop,
    })
}

/// Shared state of a run. Pages run concurrently on one task, so the database
/// and embeddings are only borrowed between awaits, never across one.
struct Run<'r, 'e, F, L, E: Embedder> {
    db: RefCell<&'r mut Db>,
    fetcher: &'r F,
    llm: &'r L,
    embeddings: RefCell<Option<&'r mut Embeddings<'e, E>>>,
    config: &'r Config,
    lang: &'r LangMode,
    domains: DomainLimiter,
    progress: &'r Progress,
    /// Pages in a row that failed at the model.
    model_failures: Cell<usize>,
    fetch_secs: Cell<f64>,
    model_secs: Cell<f64>,
    pages_cut: Cell<usize>,
    model_pages: Cell<usize>,
    /// Tokens and cost already spent when the pages started, e.g. on filling
    /// in missing vectors; not part of what a page costs.
    spent_before: Cell<(u64, f64)>,
    spent_elsewhere: (u64, f64),
}

/// After this many pages in a row fail at the model, the problem is most
/// likely the model or its settings rather than the pages, so the run stops.
/// The pages keep their failed stubs, so a run can't get stuck on pages that
/// really are bad; --retry-failed brings them back after a settings fix.
const MODEL_FAILURE_STREAK: usize = 5;

impl<F: Fetcher, L: Llm, E: Embedder> Run<'_, '_, F, L, E> {
    /// Whether starting another page could cross a budget limit: the tokens
    /// and cost so far, plus what the `in_flight` pages will likely add at
    /// the average of the `finished` ones.
    fn over_budget(&self, budget: &Budget, in_flight: usize, finished: usize) -> Option<String> {
        let (tokens, cost) = self.spent();
        let (tokens_before, cost_before) = self.spent_before.get();
        // What the running pages will likely add, at the average of the
        // finished ones, not counting what was spent before pages started.
        let still_running = |used: f64| match finished {
            0 => 0.0,
            n => used / n as f64 * in_flight as f64,
        };
        let (tokens_by_pages, cost_by_pages) = ((tokens - tokens_before) as f64, cost - cost_before);
        if let Some(max) = budget.max_tokens {
            if tokens >= max {
                return Some(format!("the token limit was reached ({tokens} of {max})"));
            }
            let coming = still_running(tokens_by_pages);
            if tokens as f64 + coming >= max as f64 {
                return Some(format!(
                    "the token limit would be reached by the pages already running ({tokens} used, about {coming:.0} \
                     more coming, limit {max})"
                ));
            }
        }
        if let Some(max) = budget.max_cost {
            if cost >= max {
                return Some(format!("the cost limit was reached (${cost:.2} of ${max:.2})"));
            }
            let coming = still_running(cost_by_pages);
            if cost + coming >= max {
                return Some(format!(
                    "the cost limit would be reached by the pages already running (${cost:.2} spent, about \
                     ${coming:.2} more coming, limit ${max:.2})"
                ));
            }
        }
        None
    }

    /// Tokens and cost of the run so far, summaries and embeddings together.
    fn spent(&self) -> (u64, f64) {
        let summaries = self.llm.usage();
        let embeddings = self
            .embeddings
            .borrow()
            .as_deref()
            .map(|e| e.embedder.usage())
            .unwrap_or_default();
        let cost = self
            .config
            .llm
            .prices()
            .cost(summaries.input_tokens, summaries.output_tokens)
            + embeddings.input_tokens as f64 * self.config.embeddings.price_per_mtok / 1e6;
        let (other_tokens, other_cost) = self.spent_elsewhere;
        (
            summaries.total_tokens() + embeddings.total_tokens() + other_tokens,
            cost + other_cost,
        )
    }

    async fn process(&self, page: &PendingPage) -> Outcome {
        match self.try_process(page).await {
            Ok(outcome) => outcome,
            Err(err) => Outcome::Fatal(err),
        }
    }

    /// `Err` is only for problems that affect every page.
    async fn try_process(&self, page: &PendingPage) -> Result<Outcome> {
        let fetched = {
            let _permit = self.domains.acquire(&page.url).await;
            let started = std::time::Instant::now();
            let fetched = self.fetcher.fetch(&page.url).await;
            self.fetch_secs
                .set(self.fetch_secs.get() + started.elapsed().as_secs_f64());
            fetched
        };
        let fetched = match fetched {
            Ok(fetched) => fetched,
            Err(err) if err.kind.is_unreachable() => {
                self.save_stub(page, Stub::Unreachable, err.kind.as_str(), &err.message, None)?;
                return Ok(Outcome::Unreachable(err.kind.as_str(), err.message));
            }
            Err(err) => {
                self.save_stub(page, Stub::Failed, err.kind.as_str(), &err.message, None)?;
                return Ok(Outcome::Failed(err.kind.as_str(), err.message));
            }
        };

        let extracted = extract(&fetched.html, &fetched.final_url, self.config.llm.max_input_chars);
        // The page's own title, as the tab shows it: the one saved with the
        // link if there is one, else the HTML <title>.
        let html_page_title = html_title(&fetched.html);
        let page_title = page
            .browser_title
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .or_else(|| html_page_title.clone());
        if extracted.truncated {
            self.pages_cut.set(self.pages_cut.get() + 1);
        }
        let title = [Some(extracted.title.as_str()), page.browser_title.as_deref()]
            .into_iter()
            .flatten()
            .find(|t| !t.trim().is_empty())
            .unwrap_or("");
        let detected_lang = whatlang::detect(&extracted.text)
            .filter(|info| info.is_reliable())
            .map(|info| info.lang().eng_name());

        let vocabulary = self.vocabulary(title, &extracted.text).await?;
        let request = SummaryRequest {
            url: &page.url,
            title,
            text: &extracted.text,
            lang: self.lang,
            detected_lang,
            vocabulary: &vocabulary,
            tags: &self.config.tags,
        };
        self.model_pages.set(self.model_pages.get() + 1);
        let started = std::time::Instant::now();
        let result = self.llm.summarize(&request).await;
        self.model_secs
            .set(self.model_secs.get() + started.elapsed().as_secs_f64());
        let summary = match result {
            Ok(summary) => {
                self.model_failures.set(0);
                summary
            }
            Err(err) => {
                let Some(rejected) = err.downcast_ref::<PageRejected>() else {
                    return Err(err.context("LLM request failed"));
                };
                let message = format!("{err:#}");
                self.save_stub(
                    page,
                    Stub::Failed,
                    rejected.kind,
                    &message,
                    html_page_title.as_deref(),
                )?;
                let streak = self.model_failures.get() + 1;
                self.model_failures.set(streak);
                if streak >= MODEL_FAILURE_STREAK {
                    let stop = anyhow::anyhow!(
                        "{streak} pages in a row failed at the model, the last one with: {message}. This \
                         usually means a problem with the model or its settings rather than with the pages, \
                         e.g. a parameter the server doesn't accept, or a model that has stopped responding. \
                         Fix it and run again with --retry-failed to retry the failed pages"
                    );
                    return Ok(Outcome::FailedAndStop(rejected.kind, message, stop));
                }
                return Ok(Outcome::Failed(rejected.kind, message));
            }
        };

        // Resolving and saving happen without an await in between, so two
        // pages can't create the same tag twice.
        let title = summary.title.trim();
        let text = summary.summary.trim();
        let created = {
            let mut db = self.db.borrow_mut();
            let ResolvedTags { page_tags, created } = resolve_tags(&db, &summary, &self.config.tags)?;
            let language = summary.language.trim().to_ascii_lowercase();
            let result = PageResult {
                title,
                summary: text,
                lang: (!language.is_empty()).then_some(language.as_str()),
                tags: &page_tags,
            };
            db.save_result(page.id, &result).context("saving result")?;
            if let Some(page_title) = &page_title {
                db.set_page_title(page.id, page_title)?;
            }
            created
        };
        self.store_vectors(page.id, title, text, created).await?;
        Ok(Outcome::Done(title.to_string()))
    }

    /// Saves a stub note for a page without a summary, so every tab still
    /// shows up in the archive: a title, a sentence saying why, and a
    /// reserved status tag.
    /// `html_title` is the page's own `<title>`, when the page loaded.
    fn save_stub(
        &self,
        page: &PendingPage,
        stub: Stub,
        kind: &str,
        message: &str,
        html_title: Option<&str>,
    ) -> Result<()> {
        let (tag_name, tag_description, not_what) = match stub {
            Stub::Unreachable => (UNREACHABLE_TAG, "Pages that could not be loaded", "loaded"),
            Stub::Failed => (
                FAILED_TAG,
                "Pages that loaded but could not be summarized",
                "summarized",
            ),
        };
        let saved = page
            .browser_title
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty());
        let (title, title_note) = match (saved, html_title) {
            (Some(saved), _) => (saved.to_string(), "The title is the one saved with the link."),
            (None, Some(own)) => (own.to_string(), "The title is the page's own."),
            (None, None) => (
                page.url
                    .split_once("://")
                    .map_or(page.url.as_str(), |(_, rest)| rest)
                    .to_string(),
                "The title is the page's address.",
            ),
        };
        let summary =
            format!("This page could not be {not_what} ({message}), so there is no summary. {title_note}");

        let mut db = self.db.borrow_mut();
        let tag = match db.find_tag(tag_name)? {
            Some((id, _)) => id,
            None => db.create_tag(tag_name, Some(tag_description))?,
        };
        let tags = [(tag_name.to_string(), tag)];
        let result = PageResult {
            title: &title,
            summary: &summary,
            lang: None,
            tags: &tags,
        };
        match stub {
            Stub::Unreachable => db.save_unreachable(page.id, &result, kind, message)?,
            Stub::Failed => db.save_failed(page.id, &result, kind, message)?,
        }
        if let Some(own) = saved.or(html_title) {
            db.set_page_title(page.id, own)?;
        }
        Ok(())
    }

    /// The existing tags to show the model: the most similar to the page, or
    /// the most used ones without embeddings.
    async fn vocabulary(&self, title: &str, text: &str) -> Result<String> {
        // Tags whose vectors failed on an earlier page go along with this
        // request, so this page can already be shown them.
        let request = {
            let mut embeddings = self.embeddings.borrow_mut();
            match embeddings.as_deref_mut() {
                Some(e) if e.index.len() > 0 || !e.unindexed_tags.is_empty() => {
                    let retried = std::mem::take(&mut e.unindexed_tags);
                    let mut texts: Vec<String> = retried.iter().map(|(_, text)| text.clone()).collect();
                    texts.push(format!("{title}\n{}", truncate_chars(text, e.page_chars)));
                    Some((e.embedder, retried, texts))
                }
                _ => None,
            }
        };
        let mut page_vector = None;
        if let Some((embedder, retried, texts)) = request {
            let result = embedder.embed(&texts).await;
            let mut embeddings = self.embeddings.borrow_mut();
            let e = embeddings.as_deref_mut().expect("embeddings were checked above");
            match result {
                Ok(mut vectors) => {
                    page_vector = vectors.pop();
                    let db = self.db.borrow();
                    for ((id, _), vector) in retried.into_iter().zip(vectors) {
                        db.put_embedding(EmbeddingKind::Tag, id, embedder.model(), &vector)?;
                        e.index.insert(id, vector);
                    }
                }
                Err(err) => {
                    e.unindexed_tags.extend(retried);
                    e.note_error(
                        self.progress,
                        "embedding a page",
                        "that page is shown the most used tags instead of the most relevant ones",
                        &err,
                    );
                }
            }
        }

        let db = self.db.borrow();
        let counted = count_tree(&db.tags()?, &db.tag_links()?);
        let limit = self.config.tags.vocabulary_limit;
        let shown = match (&page_vector, self.embeddings.borrow().as_deref()) {
            (Some(vector), Some(e)) => relevant_tags(&counted, &e.index, vector, limit),
            _ => most_used(&counted, limit),
        };
        Ok(format_vocabulary(&shown))
    }

    /// Stores the page's vector (for the end-of-run tag reconciliation) and
    /// its new tags' vectors, so the next pages can be shown them. The page is
    /// already saved, so a failure only costs vectors: the tags are retried
    /// with the next page, and the next run fills in anything still missing.
    async fn store_vectors(
        &self,
        page_id: i64,
        title: &str,
        text: &str,
        created: Vec<(i64, String)>,
    ) -> Result<()> {
        let (embedder, tags) = {
            let mut embeddings = self.embeddings.borrow_mut();
            let Some(e) = embeddings.as_deref_mut() else {
                return Ok(());
            };
            let mut tags = std::mem::take(&mut e.unindexed_tags);
            tags.extend(created);
            (e.embedder, tags)
        };
        let mut texts: Vec<String> = tags.iter().map(|(_, text)| text.clone()).collect();
        texts.push(page_embedding_text(title, text));
        let result = embedder.embed(&texts).await;

        let mut embeddings = self.embeddings.borrow_mut();
        let e = embeddings.as_deref_mut().expect("embeddings were checked above");
        let mut vectors = match result {
            Ok(vectors) => vectors,
            Err(err) => {
                e.unindexed_tags.extend(tags);
                e.note_error(
                    self.progress,
                    "storing vectors for new tags",
                    "the tags are retried with the next page, and the next run fills in missing vectors",
                    &err,
                );
                return Ok(());
            }
        };
        let db = self.db.borrow();
        let model = embedder.model();
        db.put_embedding(
            EmbeddingKind::Page,
            page_id,
            model,
            &vectors.pop().expect("one vector per text"),
        )?;
        for ((id, _), vector) in tags.into_iter().zip(vectors) {
            db.put_embedding(EmbeddingKind::Tag, id, model, &vector)?;
            e.index.insert(id, vector);
        }
        Ok(())
    }
}

/// The `limit` tags most similar to the page, among tags in use.
fn relevant_tags<'c>(
    counted: &'c [CountedTag],
    index: &TagIndex,
    page: &[f32],
    limit: usize,
) -> Vec<&'c CountedTag> {
    let by_id: HashMap<i64, &CountedTag> = counted
        .iter()
        .filter(|t| is_offerable(t))
        .map(|t| (t.id, t))
        .collect();
    index
        .nearest(page, limit, |id| by_id.contains_key(&id))
        .iter()
        .map(|id| by_id[id])
        .collect()
}

struct ResolvedTags {
    /// (raw tag as the model wrote it, tag id) for the page.
    page_tags: Vec<(String, i64)>,
    /// (tag id, text to embed) for each tag created for this page.
    created: Vec<(i64, String)>,
}

/// Normalizes the model's tags, maps spelling variants to existing tags,
/// enforces the per-page limits, and creates new tags with their
/// descriptions.
fn resolve_tags(db: &Db, summary: &PageSummary, limits: &TagConfig) -> Result<ResolvedTags> {
    let descriptions: HashMap<String, &str> = summary
        .new_tags
        .iter()
        .filter_map(|t| {
            let desc = t.description.trim();
            if desc.is_empty() {
                return None;
            }
            Some((normalize_name(&t.name)?, desc))
        })
        .collect();

    let mut seen = HashSet::new();
    let mut resolved = Vec::new();
    let mut created = Vec::new();
    for raw in &summary.tags {
        if resolved.len() >= limits.max_per_page {
            break;
        }
        let Some(name) = normalize_name(raw) else { continue };
        let id = match db.find_tag(&name)? {
            Some((id, under_other_name)) => {
                if under_other_name {
                    db.add_alias(&name, id, "rule")?;
                }
                id
            }
            None => {
                // Over the new-tag limit: skip, unless the page would end up untagged.
                if created.len() >= limits.max_new_per_page && !resolved.is_empty() {
                    continue;
                }
                // Models often leave tags out of new_tags. Naming the page the tag
                // was first used for still tells later pages what it means, e.g.
                // that `rust` was the programming language, not corrosion.
                let description = match descriptions.get(&name) {
                    Some(d) => d.to_string(),
                    None => format!("first used for: {}", summary.title.trim()),
                };
                let id = db.create_tag(&name, Some(&description))?;
                created.push((id, tag_embedding_text(&name, Some(&description))));
                id
            }
        };
        if seen.insert(id) {
            resolved.push((raw.clone(), id));
        }
    }
    Ok(ResolvedTags {
        page_tags: resolved,
        created,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use anyhow::bail;

    use std::collections::BTreeMap;

    use super::*;
    use crate::fetch::{FetchError, FetchErrorKind, FetchedPage};
    use crate::llm::NewTag;

    struct FakeFetcher;

    impl Fetcher for FakeFetcher {
        async fn fetch(&self, url: &str) -> Result<FetchedPage, FetchError> {
            // Lets other pages run, as a real fetch would.
            tokio::task::yield_now().await;
            let fail = |kind, message: &str| {
                Err(FetchError {
                    kind,
                    message: message.to_string(),
                })
            };
            if url.contains("broken") {
                return fail(FetchErrorKind::NotFound, "HTTP 404 Not Found");
            }
            if url.ends_with(".pdf") {
                return fail(
                    FetchErrorKind::UnsupportedType,
                    "unsupported content type: application/pdf",
                );
            }
            let sentence = if url.contains("bread") {
                "Sourdough bread is leavened with a starter of wild yeast and flour. "
            } else {
                "Rust is a systems programming language focused on safety and speed. "
            };
            Ok(FetchedPage {
                final_url: url.to_string(),
                html: format!(
                    "<html><head><title>Page {url}</title></head><body><article><p>{}</p></article></body></html>",
                    sentence.repeat(10)
                ),
            })
        }
    }

    /// Returns queued replies in order and records the vocabulary it was shown.
    struct FakeLlm {
        replies: RefCell<Vec<Result<PageSummary, &'static str>>>,
        seen_vocab: RefCell<Vec<String>>,
    }

    impl FakeLlm {
        fn new(replies: Vec<Result<PageSummary, &'static str>>) -> Self {
            Self {
                replies: RefCell::new(replies),
                seen_vocab: RefCell::default(),
            }
        }
    }

    impl Llm for FakeLlm {
        async fn summarize(&self, request: &SummaryRequest<'_>) -> Result<PageSummary> {
            self.seen_vocab.borrow_mut().push(request.vocabulary.to_string());
            match self.replies.borrow_mut().remove(0) {
                Ok(s) => Ok(s),
                Err("reject") => Err(PageRejected::rejected("too long".into()).into()),
                Err(other) => bail!("{other}"),
            }
        }

        /// 80 input and 20 output tokens per request.
        fn usage(&self) -> crate::usage::StageUsage {
            let requests = self.seen_vocab.borrow().len() as u64;
            crate::usage::StageUsage {
                requests,
                input_tokens: 80 * requests,
                output_tokens: 20 * requests,
                ..Default::default()
            }
        }
    }

    /// Bag-of-words vectors: texts sharing words are similar.
    struct FakeEmbedder;

    impl Embedder for FakeEmbedder {
        fn model(&self) -> &str {
            "fake"
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| {
                    let mut v = vec![0.0f32; 32];
                    for word in text
                        .to_lowercase()
                        .split(|c: char| !c.is_alphanumeric())
                        .filter(|w| w.len() > 2)
                    {
                        let h = word
                            .bytes()
                            .fold(7u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32));
                        v[h as usize % 32] += 1.0;
                    }
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                    v.into_iter().map(|x| x / norm).collect()
                })
                .collect())
        }
    }

    /// Works while a run is being prepared, then fails the chosen requests:
    /// embedding a page before the model call ("query"), or storing a
    /// finished page's vectors ("store"). Store requests number from 0;
    /// those in `failing_stores` fail.
    struct FlakyEmbedder {
        fail_query: bool,
        failing_stores: std::ops::Range<usize>,
        stores: std::cell::Cell<usize>,
    }

    impl FlakyEmbedder {
        fn new(fail_query: bool, failing_stores: std::ops::Range<usize>) -> Self {
            Self {
                fail_query,
                failing_stores,
                stores: Default::default(),
            }
        }
    }

    impl Embedder for FlakyEmbedder {
        fn model(&self) -> &str {
            "fake"
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            // The fake fetcher's pages are titled "Page <url>"; a query ends
            // with the page, after any tags being retried.
            let is_query = texts.last().is_some_and(|t| t.starts_with("Page "));
            let is_probe = texts.len() == 1 && texts[0] == "tabkeeper";
            let fails = if is_probe {
                false
            } else if is_query {
                self.fail_query
            } else {
                let n = self.stores.get();
                self.stores.set(n + 1);
                self.failing_stores.contains(&n)
            };
            if fails {
                bail!("embeddings server returned HTTP 500");
            }
            FakeEmbedder.embed(texts).await
        }
    }

    fn summary(title: &str, tags: &[&str], new: &[(&str, &str)]) -> PageSummary {
        PageSummary {
            title: title.into(),
            summary: "A summary.".into(),
            language: "EN".into(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
            new_tags: new
                .iter()
                .map(|(n, d)| NewTag {
                    name: n.to_string(),
                    description: d.to_string(),
                })
                .collect(),
        }
    }

    fn db_with(urls: &[&str]) -> Db {
        let db = Db::open_in_memory().unwrap();
        for url in urls {
            db.add_page(url, url, None, "import").unwrap();
        }
        db
    }

    async fn run(db: &mut Db, llm: &FakeLlm) -> Result<Stats> {
        process_pending(
            db,
            &FakeFetcher,
            llm,
            None::<&mut Embeddings<FakeEmbedder>>,
            &RunOptions {
                config: &Config::default(),
                lang: &LangMode::English,
                filter: &Filter::default(),
                limit: None,
                budget: Budget::default(),
                progress_to_stderr: false,
                spent_elsewhere: (0, 0.0),
            },
        )
        .await
        .and_then(RunSummary::into_result)
    }

    async fn run_with_embeddings(db: &mut Db, llm: &FakeLlm, config: &Config) -> Result<Stats> {
        let mut embeddings = Embeddings::prepare(db, &FakeEmbedder, 2000).await.unwrap();
        process_pending(
            db,
            &FakeFetcher,
            llm,
            Some(&mut embeddings),
            &RunOptions {
                config,
                lang: &LangMode::English,
                filter: &Filter::default(),
                limit: None,
                budget: Budget::default(),
                progress_to_stderr: false,
                spent_elsewhere: (0, 0.0),
            },
        )
        .await
        .and_then(RunSummary::into_result)
    }

    fn page_tag_names(db: &Db) -> Vec<String> {
        let tags = db.tags().unwrap();
        let mut names: Vec<String> = db
            .tag_links()
            .unwrap()
            .iter()
            .map(|(_, id)| tags.iter().find(|t| t.id == *id).unwrap().name.clone())
            .collect();
        names.sort();
        names
    }

    #[tokio::test]
    async fn processes_pages_and_grows_vocabulary() {
        let mut db = db_with(&["https://a.com/", "https://broken.com/", "https://b.com/"]);
        let llm = FakeLlm::new(vec![
            Ok(summary(
                "Rust intro",
                &["Rust Programming", "technology/Memory Safety"],
                &[("rust-programming", "The Rust language")],
            )),
            Ok(summary(
                "More Rust",
                &["rust-programming", "status/unreachable"],
                &[],
            )),
        ]);
        let stats = run(&mut db, &llm).await.unwrap();
        assert_eq!(
            stats,
            Stats {
                done: 2,
                unreachable: 1,
                causes: BTreeMap::from([("not_found".into(), 1)]),
                model_pages: 2,
                ..Stats::default()
            }
        );

        // The second page was shown the tags the first page created.
        let seen = llm.seen_vocab.borrow();
        assert_eq!(seen[0], "");
        assert_eq!(
            seen[1],
            "memory-safety (1) - first used for: Rust intro\nrust-programming (1) - The Rust language"
        );

        // The model's status/ tag was dropped and paths became flat names; the
        // only status tag is the one tabkeeper put on the unreachable page,
        // which the model is never offered.
        assert_eq!(
            page_tag_names(&db),
            [
                "memory-safety",
                "rust-programming",
                "rust-programming",
                "status/unreachable"
            ]
        );
        assert!(!seen[1].contains("status/"));
        let pages = db.note_pages().unwrap();
        assert_eq!(pages[0].lang.as_deref(), Some("en"));
        assert_eq!(db.status_counts().unwrap().unreachable, 1);

        // The unreachable page got a stub note saying why.
        let stub = pages.iter().find(|p| p.url.contains("broken")).unwrap();
        assert_eq!(stub.title, "broken.com/");
        assert_eq!(
            stub.summary,
            "This page could not be loaded (HTTP 404 Not Found), so there is no summary. The title is the page's address."
        );
    }

    #[tokio::test]
    async fn pages_that_load_but_cant_be_used_get_failed_stubs() {
        let mut db = db_with(&["https://a.com/paper.pdf", "https://b.com/"]);
        let llm = FakeLlm::new(vec![Err("reject")]);
        let stats = run(&mut db, &llm).await.unwrap();
        assert_eq!(
            stats,
            Stats {
                failed: 2,
                causes: BTreeMap::from([("rejected".into(), 1), ("unsupported_type".into(), 1)]),
                model_pages: 1,
                ..Stats::default()
            }
        );
        let kinds = db.failure_kinds().unwrap();
        assert_eq!(
            kinds,
            [("rejected".to_string(), 1), ("unsupported_type".to_string(), 1)]
        );

        // Both still get a note, so no tab is lost from the archive.
        assert_eq!(page_tag_names(&db), ["status/failed", "status/failed"]);
        let pages = db.note_pages().unwrap();
        assert_eq!(
            pages[0].summary,
            "This page could not be summarized (unsupported content type: application/pdf), so there is no \
             summary. The title is the page's address."
        );
        assert!(
            pages[1]
                .summary
                .starts_with("This page could not be summarized (too long)"),
            "{}",
            pages[1].summary
        );
        // The page loaded, so its stub has the page's own title, not the address.
        assert_eq!(pages[1].title, "Page https://b.com/");
        assert_eq!(pages[1].page_title.as_deref(), Some("Page https://b.com/"));
        assert!(
            pages[1].summary.ends_with("The title is the page's own."),
            "{}",
            pages[1].summary
        );
    }

    #[tokio::test]
    async fn retried_stub_that_fails_again_is_rewritten() {
        // An earlier run found the page unreachable and wrote its stub note.
        let mut db = db_with(&["https://a.com/paper.pdf"]);
        let page = db.pending_pages().unwrap()[0].id;
        let tag = db.create_tag(UNREACHABLE_TAG, None).unwrap();
        let tags = [(UNREACHABLE_TAG.to_string(), tag)];
        let stub = PageResult {
            title: "Paper",
            summary: "Unreachable.",
            lang: None,
            tags: &tags,
        };
        db.save_unreachable(page, &stub, "timeout", "timed out").unwrap();
        let dir = tempfile::tempdir().unwrap();
        crate::render::render_all(&db, dir.path(), crate::config::TitleStyle::Both).unwrap();

        // This time it loads, but it's a PDF.
        db.retry_failed().unwrap();
        run(&mut db, &FakeLlm::new(vec![])).await.unwrap();
        crate::render::render_all(&db, dir.path(), crate::config::TitleStyle::Both).unwrap();
        let note = std::fs::read_to_string(dir.path().join("notes/paper.md")).unwrap();
        assert!(
            note.contains("could not be summarized (unsupported content type"),
            "{note}"
        );
        assert!(
            note.ends_with("#status/failed\n"),
            "the old status tag is gone: {note}"
        );
        assert_eq!(db.failure_kinds().unwrap(), [("unsupported_type".to_string(), 1)]);
    }

    #[tokio::test]
    async fn keeps_the_page_title_from_the_link_or_the_html() {
        let db = Db::open_in_memory().unwrap();
        db.add_page("https://a.com/", "https://a.com/", None, "import")
            .unwrap();
        db.add_page(
            "https://b.com/",
            "https://b.com/",
            Some("Saved Tab Title"),
            "import",
        )
        .unwrap();
        let mut db = db;
        let llm = FakeLlm::new(vec![
            Ok(summary("Model A", &["a"], &[])),
            Ok(summary("Model B", &["b"], &[])),
        ]);
        run(&mut db, &llm).await.unwrap();
        let titles: Vec<(String, Option<String>)> = db
            .note_pages()
            .unwrap()
            .into_iter()
            .map(|p| (p.title, p.page_title))
            .collect();
        assert_eq!(
            titles,
            [
                ("Model A".to_string(), Some("Page https://a.com/".to_string())),
                ("Model B".to_string(), Some("Saved Tab Title".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn browser_title_is_used_for_stub_notes() {
        let db = Db::open_in_memory().unwrap();
        db.add_page(
            "https://broken.com/x",
            "https://broken.com/x",
            Some("My saved tab"),
            "import",
        )
        .unwrap();
        let mut db = db;
        run(&mut db, &FakeLlm::new(vec![])).await.unwrap();
        let stub = &db.note_pages().unwrap()[0];
        assert_eq!(stub.title, "My saved tab");
        assert!(
            stub.summary
                .ends_with("The title is the one saved with the link.")
        );
        // Retrying makes it pending again.
        assert_eq!(db.retry_failed().unwrap(), 1);
        assert_eq!(db.pending_pages().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn filter_leaves_excluded_pages_pending() {
        let mut db = db_with(&["https://a.com/", "https://skip.example.com/"]);
        let llm = FakeLlm::new(vec![Ok(summary("A", &["alpha"], &[]))]);
        let config = crate::config::FilterConfig {
            mode: crate::config::FilterMode::Deny,
            rules: vec!["domain:example.com".into()],
        };
        let filter = Filter::new(&config, &[], &[]).unwrap();
        let stats = process_pending(
            &mut db,
            &FakeFetcher,
            &llm,
            None::<&mut Embeddings<FakeEmbedder>>,
            &RunOptions {
                config: &Config::default(),
                lang: &LangMode::English,
                filter: &filter,
                limit: None,
                budget: Budget::default(),
                progress_to_stderr: false,
                spent_elsewhere: (0, 0.0),
            },
        )
        .await
        .unwrap()
        .into_result()
        .unwrap();
        assert_eq!(
            stats,
            Stats {
                done: 1,
                filtered: 1,
                model_pages: 1,
                ..Stats::default()
            }
        );
        assert_eq!(db.pending_pages().unwrap()[0].url, "https://skip.example.com/");
    }

    #[tokio::test]
    async fn concurrent_pages_share_one_tag_and_all_finish() {
        let urls: Vec<String> = (0..12).map(|i| format!("https://site{i}.com/")).collect();
        let mut db = db_with(&urls.iter().map(String::as_str).collect::<Vec<_>>());
        let replies = (0..12)
            .map(|i| {
                Ok(summary(
                    &format!("Page {i}"),
                    &["shared-topic", &format!("own-{i}")],
                    &[],
                ))
            })
            .collect();
        let llm = FakeLlm::new(replies);
        let mut config = Config::default();
        config.run.sequential_start = 2;
        config.run.concurrency = 5;
        let stats = process_pending(
            &mut db,
            &FakeFetcher,
            &llm,
            None::<&mut Embeddings<FakeEmbedder>>,
            &RunOptions {
                config: &config,
                lang: &LangMode::English,
                filter: &Filter::default(),
                limit: None,
                budget: Budget::default(),
                progress_to_stderr: false,
                spent_elsewhere: (0, 0.0),
            },
        )
        .await
        .unwrap()
        .into_result()
        .unwrap();
        assert_eq!(
            stats,
            Stats {
                done: 12,
                model_pages: 12,
                ..Stats::default()
            }
        );
        let shared: Vec<_> = db
            .tags()
            .unwrap()
            .into_iter()
            .filter(|t| t.name == "shared-topic")
            .collect();
        assert_eq!(shared.len(), 1);
        assert_eq!(
            db.tag_links()
                .unwrap()
                .iter()
                .filter(|(_, id)| *id == shared[0].id)
                .count(),
            12
        );
    }

    #[tokio::test]
    async fn merges_spelling_variants_into_existing_tags() {
        let mut db = db_with(&["https://a.com/", "https://b.com/"]);
        let llm = FakeLlm::new(vec![
            Ok(summary("T", &["board-games", "machine-learning", "glass"], &[])),
            Ok(summary("T", &["Board Games", "machinelearning", "glasses"], &[])),
        ]);
        run(&mut db, &llm).await.unwrap();
        // Spacing and hyphen variants resolve to the existing tags; a plural is
        // a new tag until reconciliation decides whether it means the same.
        let names: Vec<String> = db.tags().unwrap().into_iter().map(|t| t.name).collect();
        assert_eq!(names, ["board-games", "glass", "glasses", "machine-learning"]);
        // The variant is recorded, so the next page finds it directly.
        assert_eq!(
            db.find_tag("machinelearning").unwrap().map(|(_, other)| other),
            Some(true)
        );
    }

    #[tokio::test]
    async fn enforces_new_tag_limit() {
        let mut db = db_with(&["https://a.com/"]);
        let llm = FakeLlm::new(vec![Ok(summary("T", &["a", "b", "c", "d", "e"], &[]))]);
        run(&mut db, &llm).await.unwrap();
        assert_eq!(page_tag_names(&db), ["a", "b", "c"]);
    }

    #[tokio::test]
    async fn embeddings_pick_the_most_relevant_tags() {
        let mut db = db_with(&[
            "https://a.com/rust",
            "https://b.com/bread",
            "https://c.com/bread-again",
        ]);
        let llm = FakeLlm::new(vec![
            Ok(summary(
                "Rust",
                &["rust-programming", "memory-safety", "systems-programming"],
                &[(
                    "rust-programming",
                    "Rust is a systems programming language focused on safety",
                )],
            )),
            Ok(summary(
                "Bread",
                &["sourdough", "baking"],
                &[("sourdough", "Sourdough bread leavened with wild yeast")],
            )),
            Ok(summary("Bread again", &["sourdough"], &[])),
        ]);
        let config = Config {
            tags: TagConfig {
                vocabulary_limit: 1,
                ..TagConfig::default()
            },
            ..Config::default()
        };
        run_with_embeddings(&mut db, &llm, &config).await.unwrap();

        // With room for one tag, the bread page was shown sourdough, not a rust tag.
        let seen = llm.seen_vocab.borrow();
        assert!(seen[2].starts_with("sourdough (1)"), "{}", seen[2]);

        // Every tag and every page has a vector now.
        assert!(db.tags_missing_embedding("fake").unwrap().is_empty());
        assert!(db.pages_missing_embedding("fake").unwrap().is_empty());
        assert_eq!(db.embeddings(EmbeddingKind::Page, "fake").unwrap().len(), 3);
    }

    #[tokio::test]
    async fn failed_page_embedding_falls_back_to_most_used_tags() {
        let mut db = db_with(&["https://a.com/", "https://b.com/"]);
        let llm = FakeLlm::new(vec![
            Ok(summary("First", &["alpha"], &[])),
            Ok(summary("Second", &["beta"], &[])),
        ]);
        let embedder = FlakyEmbedder::new(true, 0..0);
        let mut embeddings = Embeddings::prepare(&db, &embedder, 2000).await.unwrap();
        let stats = process_pending(
            &mut db,
            &FakeFetcher,
            &llm,
            Some(&mut embeddings),
            &RunOptions {
                config: &Config::default(),
                lang: &LangMode::English,
                filter: &Filter::default(),
                limit: None,
                budget: Budget::default(),
                progress_to_stderr: false,
                spent_elsewhere: (0, 0.0),
            },
        )
        .await
        .unwrap()
        .into_result()
        .unwrap();
        assert_eq!(
            stats,
            Stats {
                done: 2,
                model_pages: 2,
                ..Stats::default()
            }
        );
        assert_eq!(embeddings.errors, 1);
        // The second page couldn't be embedded, so it was shown the most used tags.
        assert!(llm.seen_vocab.borrow()[1].starts_with("alpha (1)"));
    }

    #[tokio::test]
    async fn failed_vector_storage_keeps_the_page_and_is_backfilled() {
        let mut db = db_with(&["https://a.com/", "https://b.com/"]);
        let llm = FakeLlm::new(vec![
            Ok(summary("First", &["alpha"], &[])),
            Ok(summary("Second", &["beta"], &[])),
        ]);
        let embedder = FlakyEmbedder::new(false, 0..usize::MAX);
        let mut embeddings = Embeddings::prepare(&db, &embedder, 2000).await.unwrap();
        let stats = process_pending(
            &mut db,
            &FakeFetcher,
            &llm,
            Some(&mut embeddings),
            &RunOptions {
                config: &Config::default(),
                lang: &LangMode::English,
                filter: &Filter::default(),
                limit: None,
                budget: Budget::default(),
                progress_to_stderr: false,
                spent_elsewhere: (0, 0.0),
            },
        )
        .await
        .unwrap()
        .into_result()
        .unwrap();
        assert_eq!(
            stats,
            Stats {
                done: 2,
                model_pages: 2,
                ..Stats::default()
            }
        );
        assert_eq!(embeddings.errors, 2);
        // alpha was retried with the second page's query, which worked; beta's
        // store failed on the last page, and so did both page vectors.
        let missing: Vec<String> = db
            .tags_missing_embedding("fake")
            .unwrap()
            .into_iter()
            .map(|(_, text)| text)
            .collect();
        assert_eq!(missing, ["beta: first used for: Second"]);
        assert_eq!(db.pages_missing_embedding("fake").unwrap().len(), 2);
        // The next run fills in what's missing.
        Embeddings::prepare(&db, &FakeEmbedder, 2000).await.unwrap();
        assert!(db.tags_missing_embedding("fake").unwrap().is_empty());
        assert!(db.pages_missing_embedding("fake").unwrap().is_empty());
    }

    #[tokio::test]
    async fn tags_from_a_failed_store_are_shown_to_the_next_page() {
        let mut db = db_with(&[
            "https://a.com/bread",
            "https://b.com/rust",
            "https://c.com/rust-again",
        ]);
        let llm = FakeLlm::new(vec![
            Ok(summary(
                "Bread",
                &["sourdough"],
                &[("sourdough", "Sourdough bread leavened with wild yeast")],
            )),
            Ok(summary(
                "Rust",
                &["rust-programming"],
                &[(
                    "rust-programming",
                    "Rust is a systems programming language focused on safety",
                )],
            )),
            Ok(summary("Rust again", &["rust-programming"], &[])),
        ]);
        // The bread page's vectors are stored; the first rust page's store fails.
        let embedder = FlakyEmbedder::new(false, 1..2);
        let mut embeddings = Embeddings::prepare(&db, &embedder, 2000).await.unwrap();
        let config = Config {
            tags: TagConfig {
                vocabulary_limit: 1,
                ..TagConfig::default()
            },
            ..Config::default()
        };
        process_pending(
            &mut db,
            &FakeFetcher,
            &llm,
            Some(&mut embeddings),
            &RunOptions {
                config: &config,
                lang: &LangMode::English,
                filter: &Filter::default(),
                limit: None,
                budget: Budget::default(),
                progress_to_stderr: false,
                spent_elsewhere: (0, 0.0),
            },
        )
        .await
        .unwrap()
        .into_result()
        .unwrap();
        assert_eq!(embeddings.errors, 1);
        // With room for one tag, the second rust page was shown rust-programming,
        // though its vector failed to store on the page that created it.
        let seen = llm.seen_vocab.borrow();
        assert!(seen[2].starts_with("rust-programming (1)"), "{}", seen[2]);
        assert!(db.tags_missing_embedding("fake").unwrap().is_empty());
    }

    #[tokio::test]
    async fn prepare_backfills_missing_vectors() {
        let mut db = db_with(&["https://a.com/"]);
        let llm = FakeLlm::new(vec![Ok(summary("T", &["alpha", "beta"], &[]))]);
        run(&mut db, &llm).await.unwrap(); // without embeddings
        assert_eq!(db.tags_missing_embedding("fake").unwrap().len(), 2);
        let embeddings = Embeddings::prepare(&db, &FakeEmbedder, 2000).await.unwrap();
        assert_eq!(embeddings.index.len(), 2);
        assert!(db.pages_missing_embedding("fake").unwrap().is_empty());
    }

    #[tokio::test]
    async fn five_model_failures_in_a_row_stop_the_run_without_getting_stuck() {
        let urls: Vec<String> = (0..7).map(|i| format!("https://site{i}.com/")).collect();
        let mut db = db_with(&urls.iter().map(String::as_str).collect::<Vec<_>>());
        let llm = FakeLlm::new(vec![Err("reject"); 5]);
        let err = run(&mut db, &llm).await.unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("5 pages in a row failed at the model"),
            "{message}"
        );
        assert!(
            message.contains("5 failed so far"),
            "the count matches the database: {message}"
        );
        // The five keep their failed stubs, so the next run moves past them...
        let counts = db.status_counts().unwrap();
        assert_eq!((counts.failed, counts.pending), (5, 2));
        let llm = FakeLlm::new(vec![Ok(summary("F", &["f"], &[])), Ok(summary("G", &["g"], &[]))]);
        assert_eq!(
            run(&mut db, &llm).await.unwrap(),
            Stats {
                done: 2,
                model_pages: 2,
                ..Stats::default()
            }
        );
        // ...and --retry-failed brings them back once the cause is fixed.
        assert_eq!(db.retry_failed().unwrap(), 5);
    }

    #[tokio::test]
    async fn a_success_resets_the_model_failure_count() {
        let urls: Vec<String> = (0..9).map(|i| format!("https://site{i}.com/")).collect();
        let mut db = db_with(&urls.iter().map(String::as_str).collect::<Vec<_>>());
        let mut replies = vec![Err("reject"); 4];
        replies.push(Ok(summary("E", &["e"], &[])));
        replies.extend(vec![Err("reject"); 4]);
        let stats = run(&mut db, &FakeLlm::new(replies)).await.unwrap();
        assert_eq!(
            stats,
            Stats {
                done: 1,
                failed: 8,
                causes: BTreeMap::from([("rejected".into(), 8)]),
                model_pages: 9,
                ..Stats::default()
            }
        );
    }

    #[tokio::test]
    async fn a_budget_stops_starting_pages() {
        let urls: Vec<String> = (0..6).map(|i| format!("https://site{i}.com/")).collect();
        let mut db = db_with(&urls.iter().map(String::as_str).collect::<Vec<_>>());
        let llm = FakeLlm::new(
            (0..6)
                .map(|i| Ok(summary(&format!("P{i}"), &["t"], &[])))
                .collect(),
        );
        let budget = Budget {
            max_tokens: Some(250),
            max_cost: None,
        };
        let options = RunOptions {
            config: &Config::default(),
            lang: &LangMode::English,
            filter: &Filter::default(),
            limit: None,
            budget,
            progress_to_stderr: false,
            spent_elsewhere: (0, 0.0),
        };
        let summary = process_pending(
            &mut db,
            &FakeFetcher,
            &llm,
            None::<&mut Embeddings<FakeEmbedder>>,
            &options,
        )
        .await
        .unwrap();
        // 100 tokens per page: the third page reaches 300 of 250.
        assert_eq!(summary.stats.done, 3);
        assert!(
            matches!(&summary.stop, Some(Stop::Budget(reason)) if reason.contains("300 of 250")),
            "{:?}",
            summary.stop
        );
        assert_eq!(
            db.status_counts().unwrap().pending,
            3,
            "the rest is left for the next run"
        );
        assert!(summary.into_result().is_ok(), "a budget stop is not an error");
    }

    #[tokio::test]
    async fn a_budget_counts_the_pages_already_running() {
        let urls: Vec<String> = (0..8).map(|i| format!("https://site{i}.com/")).collect();
        let mut db = db_with(&urls.iter().map(String::as_str).collect::<Vec<_>>());
        let llm = FakeLlm::new(
            (0..8)
                .map(|i| Ok(summary(&format!("P{i}"), &["t"], &[])))
                .collect(),
        );
        let mut config = Config::default();
        config.run.sequential_start = 1;
        config.run.concurrency = 4;
        let options = RunOptions {
            config: &config,
            lang: &LangMode::English,
            filter: &Filter::default(),
            limit: None,
            budget: Budget {
                max_tokens: Some(350),
                max_cost: None,
            },
            progress_to_stderr: false,
            spent_elsewhere: (0, 0.0),
        };
        let summary = process_pending(
            &mut db,
            &FakeFetcher,
            &llm,
            None::<&mut Embeddings<FakeEmbedder>>,
            &options,
        )
        .await
        .unwrap();
        // After the first page (100 tokens), pages start while 100 used plus
        // 100 for each page running stays under 350: three more, not seven.
        assert_eq!(summary.stats.done, 4);
        assert!(
            matches!(&summary.stop, Some(Stop::Budget(reason)) if reason.contains("pages already running")),
            "{:?}",
            summary.stop
        );
        assert_eq!(db.status_counts().unwrap().pending, 4);
    }

    #[tokio::test]
    async fn tokens_spent_before_pages_start_dont_inflate_the_projection() {
        let urls: Vec<String> = (0..8).map(|i| format!("https://site{i}.com/")).collect();
        let mut db = db_with(&urls.iter().map(String::as_str).collect::<Vec<_>>());
        let llm = FakeLlm::new(
            (0..8)
                .map(|i| Ok(summary(&format!("P{i}"), &["t"], &[])))
                .collect(),
        );
        // 2,000 tokens already used before the first page, like filling in vectors.
        llm.seen_vocab.borrow_mut().extend((0..20).map(|_| String::new()));
        let mut config = Config::default();
        config.run.sequential_start = 1;
        config.run.concurrency = 4;
        let options = RunOptions {
            config: &config,
            lang: &LangMode::English,
            filter: &Filter::default(),
            limit: None,
            budget: Budget {
                max_tokens: Some(2_350),
                max_cost: None,
            },
            progress_to_stderr: false,
            spent_elsewhere: (0, 0.0),
        };
        let summary = process_pending(
            &mut db,
            &FakeFetcher,
            &llm,
            None::<&mut Embeddings<FakeEmbedder>>,
            &options,
        )
        .await
        .unwrap();
        // Pages cost 100 each, not 2,100: same as without the earlier 2,000.
        assert_eq!(summary.stats.done, 4);
    }

    #[tokio::test]
    async fn tokens_spent_elsewhere_count_toward_the_budget() {
        let urls: Vec<String> = (0..5).map(|i| format!("https://site{i}.com/")).collect();
        let mut db = db_with(&urls.iter().map(String::as_str).collect::<Vec<_>>());
        let llm = FakeLlm::new(
            (0..5)
                .map(|i| Ok(summary(&format!("P{i}"), &["t"], &[])))
                .collect(),
        );
        let options = RunOptions {
            config: &Config::default(),
            lang: &LangMode::English,
            filter: &Filter::default(),
            limit: None,
            budget: Budget {
                max_tokens: Some(1_150),
                max_cost: None,
            },
            progress_to_stderr: false,
            // E.g. a failed embeddings setup.
            spent_elsewhere: (1_000, 0.0),
        };
        let summary = process_pending(
            &mut db,
            &FakeFetcher,
            &llm,
            None::<&mut Embeddings<FakeEmbedder>>,
            &options,
        )
        .await
        .unwrap();
        // 1,000 + 100 per page: the second page reaches 1,200 of 1,150.
        assert_eq!(summary.stats.done, 2);
        assert!(
            matches!(&summary.stop, Some(Stop::Budget(reason)) if reason.contains("1200 of 1150")),
            "{:?}",
            summary.stop
        );
    }

    #[tokio::test]
    async fn a_cost_budget_uses_the_configured_prices() {
        let mut db = db_with(&["https://a.com/", "https://b.com/", "https://c.com/"]);
        let llm = FakeLlm::new(
            (0..3)
                .map(|i| Ok(summary(&format!("P{i}"), &["t"], &[])))
                .collect(),
        );
        let mut config = Config::default();
        // $10,000 per million input tokens: 80 tokens cost $0.80.
        config.llm.price_input_per_mtok = 10_000.0;
        let options = RunOptions {
            config: &config,
            lang: &LangMode::English,
            filter: &Filter::default(),
            limit: None,
            budget: Budget {
                max_tokens: None,
                max_cost: Some(1.5),
            },
            progress_to_stderr: false,
            spent_elsewhere: (0, 0.0),
        };
        let summary = process_pending(
            &mut db,
            &FakeFetcher,
            &llm,
            None::<&mut Embeddings<FakeEmbedder>>,
            &options,
        )
        .await
        .unwrap();
        assert_eq!(summary.stats.done, 2);
        assert!(
            matches!(&summary.stop, Some(Stop::Budget(reason)) if reason.contains("$1.60 of $1.50")),
            "{:?}",
            summary.stop
        );
    }

    #[tokio::test]
    async fn rejected_page_fails_but_server_error_stops_the_run() {
        let mut db = db_with(&["https://a.com/", "https://b.com/", "https://c.com/"]);
        let llm = FakeLlm::new(vec![Err("reject"), Err("connection refused")]);
        let err = run(&mut db, &llm).await.unwrap_err();
        assert!(format!("{err:#}").contains("connection refused"), "{err:#}");
        let counts = db.status_counts().unwrap();
        // a.com failed; b.com and c.com stay pending for the next run.
        assert_eq!((counts.failed, counts.pending), (1, 2));
    }
}

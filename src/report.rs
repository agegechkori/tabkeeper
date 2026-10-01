use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::db::Db;
use crate::import::ImportStats;
use crate::pipeline::{RunSummary, Stop};
use crate::tags::{RESERVED_PREFIX, count_tree};
use crate::usage::{StageUsage, is_local};

/// Everything a run reports. Saved as JSON in the database and printed as
/// text; `--json` prints the JSON.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Report {
    pub started_at: String,
    pub finished_at: String,
    pub duration_secs: f64,
    /// Why the run ended before all pending pages were processed.
    pub stopped: Option<String>,
    /// What the URL list contained, for `import`.
    pub import: Option<ImportStats>,
    pub run: RunCounts,
    pub archive: ArchiveCounts,
    /// (language code, pages) for done pages in the archive.
    pub languages: Vec<(String, usize)>,
    /// (domain, pages) for the archive's most common domains.
    pub domains: Vec<(String, usize)>,
    pub model: ModelReport,
    pub tabkeeper: ProcessUsage,
    pub tags: TagCounts,
    /// The tag review at the end of the run, if it ran.
    pub review: Option<ReviewReport>,
    pub warnings: Vec<String>,
    /// Every failed or unreachable page in the archive, with its error.
    pub failed_pages: Vec<FailedPage>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RunCounts {
    pub processed: usize,
    pub done: usize,
    pub unreachable: usize,
    pub failed: usize,
    /// Pending pages the filter skipped; they stay pending.
    pub skipped_by_filter: usize,
    /// Pages still pending after the run.
    pub pending: usize,
    /// Pages sent to the model (done, or failed at the model).
    pub model_pages: usize,
    /// Unreachable and failed pages by cause, e.g. `not_found`.
    pub causes: BTreeMap<String, usize>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ArchiveCounts {
    pub done: usize,
    pub unreachable: usize,
    pub failed: usize,
    pub pending: usize,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelReport {
    pub summaries: Stage,
    pub embeddings: Option<Stage>,
    /// What Ollama says about the loaded summary model, if it's Ollama.
    pub server: Option<ServerInfo>,
    /// US dollars for the whole run; `None` for a cloud model without prices.
    pub cost: Option<f64>,
    pub local: bool,
    pub pages_cut: usize,
    pub max_input_chars: usize,
    pub fetch_secs_per_page: f64,
    /// Average time from sending a summary request to its reply, including
    /// time spent queued at the server.
    pub model_secs_per_page: f64,
    /// Pages processed at the same time (`run.concurrency`).
    pub concurrency: usize,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Stage {
    pub model: String,
    pub server: String,
    /// `[llm.extra_body]` settings, e.g. `reasoning_effort=none`.
    pub options: Vec<String>,
    pub usage: StageUsage,
    /// Embedding requests that failed during the run (embeddings only).
    pub errors: usize,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerInfo {
    pub loaded_bytes: u64,
    pub gpu_bytes: u64,
    pub context_tokens: Option<u64>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcessUsage {
    pub cpu_secs: Option<f64>,
    pub peak_memory_bytes: Option<u64>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TagCounts {
    /// Topic tags on at least one page.
    pub in_use: usize,
    pub new_this_run: usize,
    pub used_once: usize,
    pub untagged_pages: usize,
    /// (tag, pages), most used first.
    pub top: Vec<(String, usize)>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FailedPage {
    pub url: String,
    pub status: String,
    pub cause: String,
    pub error: String,
}

/// What `build` needs to know about the run besides the database.
/// What the tag review proposed and what was applied.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReviewReport {
    pub model: String,
    pub usage: StageUsage,
    /// Embedding requests the review made: missing vectors and tag names.
    pub embed_usage: StageUsage,
    pub merge_candidates: usize,
    pub split_candidates: usize,
    pub proposed: usize,
    pub merges: usize,
    pub renames: usize,
    pub splits: usize,
    /// Groups of tags put in the tag tree, and how many tags they held.
    pub placements: usize,
    pub tags_placed: usize,
    /// Tags still outside the tag tree (hierarchical mode).
    pub unplaced: usize,
    pub declined: usize,
    /// Proposed changes left unapplied because nobody could be asked.
    pub not_asked: usize,
    /// The revisions applied, oldest first; `tabkeeper undo` reverts the last.
    pub revisions: Vec<i64>,
    /// Approved changes that couldn't be applied, with the reason.
    pub failed: Vec<String>,
    /// Review requests that failed; their tags are checked again next time.
    pub failed_requests: usize,
    /// Why the review didn't run or failed.
    pub error: Option<String>,
    /// Why putting tags in the tree failed, after the rest of the review.
    pub tree_error: Option<String>,
}

impl ReviewReport {
    pub fn applied(&self) -> usize {
        self.merges + self.renames + self.splits + self.placements
    }

    pub fn count_applied(&mut self, change: &crate::db::TagChange) {
        use crate::db::TagChange;
        match change {
            TagChange::Merge { .. } => self.merges += 1,
            TagChange::Rename { .. } => self.renames += 1,
            TagChange::Split { .. } => self.splits += 1,
            TagChange::Place { tags, .. } => {
                self.placements += 1;
                self.tags_placed += tags.len();
            }
        }
    }

    /// One line for the end of a run or `revise-tags`.
    pub fn line(&self) -> String {
        if let Some(error) = &self.error {
            return format!("the tag review failed ({error}); run `tabkeeper revise-tags` to try again");
        }
        let mut notes: Vec<String> = Vec::new();
        if self.unplaced > 0 {
            notes.push(format!(
                "{} not in the tag tree yet, for the next `tabkeeper revise-tags`",
                crate::reconcile::count(self.unplaced, "tag", "tags")
            ));
        }
        if let Some(error) = &self.tree_error {
            notes.push(format!("putting tags in the tree failed ({error})"));
        }
        if self.proposed == 0 {
            if notes.is_empty() {
                return "no changes needed".into();
            }
            return notes.join(" · ");
        }
        let mut done = format!(
            "{} merges · {} renames · {} splits",
            self.merges, self.renames, self.splits
        );
        if self.placements > 0 {
            done.push_str(&format!(
                " · {} placed in the tree",
                crate::reconcile::count(self.tags_placed, "tag", "tags")
            ));
        }
        let mut parts = vec![format!(
            "{} of {} proposed changes applied ({done})",
            self.applied(),
            self.proposed
        )];
        if self.declined > 0 {
            parts.push(format!("{} declined", self.declined));
        }
        if self.failed_requests > 0 {
            parts.push(format!(
                "{} review requests failed, to be retried next time",
                self.failed_requests
            ));
        }
        if !self.failed.is_empty() {
            parts.push(format!(
                "{} couldn't be applied ({})",
                self.failed.len(),
                self.failed.join("; ")
            ));
        }
        if self.not_asked > 0 {
            parts.push(format!(
                "{} waiting: run `tabkeeper revise-tags` in a terminal, or add --yes",
                self.not_asked
            ));
        }
        parts.extend(notes);
        parts.join(" · ")
    }
}

pub struct RunContext<'a> {
    pub started_at: &'a str,
    pub started: std::time::Instant,
    pub import: Option<&'a ImportStats>,
    pub summary: &'a RunSummary,
    pub summaries: StageUsage,
    pub embeddings: Option<(StageUsage, usize)>,
    pub review: Option<ReviewReport>,
    pub config: &'a Config,
}

const TOP_DOMAINS: usize = 8;
const TOP_TAGS: usize = 20;

pub async fn build(db: &Db, ctx: &RunContext<'_>) -> Result<Report> {
    let config = ctx.config;
    let stats = &ctx.summary.stats;
    let counts = db.status_counts()?;
    let processed = stats.done + stats.unreachable + stats.failed;

    let counted = count_tree(&db.tags()?, &db.tag_links()?);
    let topic: Vec<_> = counted
        .iter()
        .filter(|t| t.total > 0 && !t.name.starts_with(RESERVED_PREFIX))
        .collect();
    let mut top: Vec<(String, usize)> = topic.iter().map(|t| (t.path.clone(), t.total)).collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    top.truncate(TOP_TAGS);

    let local = is_local(&config.llm.base_url);
    let summaries = Stage {
        model: config.llm.model.clone(),
        server: config.llm.base_url.clone(),
        options: config
            .llm
            .extra_body
            .iter()
            .map(|(k, v)| format!("{k}={}", v.as_str().map_or(v.to_string(), str::to_string)))
            .collect(),
        usage: ctx.summaries.clone(),
        errors: 0,
    };
    let embeddings = ctx.embeddings.as_ref().map(|(usage, errors)| Stage {
        model: config.embeddings.model.clone(),
        server: config
            .embeddings
            .base_url
            .clone()
            .unwrap_or_else(|| config.llm.base_url.clone()),
        options: Vec::new(),
        usage: usage.clone(),
        errors: *errors,
    });
    let prices = config.llm.prices();
    let embed_tokens = embeddings.as_ref().map_or(0, |e| e.usage.input_tokens);
    // Without prices for the summaries, which cost by far the most, the cost
    // of a cloud run is unknown, whatever the embeddings cost.
    let cost = if local {
        Some(0.0)
    } else if prices.is_set() {
        Some(
            prices.cost(summaries.usage.input_tokens, summaries.usage.output_tokens)
                + embed_tokens as f64 * config.embeddings.price_per_mtok / 1e6
                // The tag review is priced like the summaries.
                + ctx.review.as_ref().map_or(0.0, |r| {
                    prices.cost(r.usage.input_tokens, r.usage.output_tokens)
                        + r.embed_usage.input_tokens as f64 * config.embeddings.price_per_mtok / 1e6
                }),
        )
    } else {
        None
    };

    let finished_at = crate::db::now();
    let mut report = Report {
        started_at: ctx.started_at.to_string(),
        finished_at: finished_at.clone(),
        duration_secs: ctx.started.elapsed().as_secs_f64(),
        stopped: ctx.summary.stop.as_ref().map(|stop| match stop {
            Stop::Error(err) => format!("{err:#}"),
            Stop::Budget(reason) => reason.clone(),
        }),
        import: ctx.import.cloned(),
        run: RunCounts {
            processed,
            done: stats.done,
            unreachable: stats.unreachable,
            failed: stats.failed,
            skipped_by_filter: stats.filtered,
            pending: counts.pending,
            model_pages: stats.model_pages,
            causes: stats.causes.clone(),
        },
        archive: ArchiveCounts {
            done: counts.done,
            unreachable: counts.unreachable,
            failed: counts.failed,
            pending: counts.pending,
        },
        languages: db.language_counts()?,
        domains: top_domains(&db.note_page_urls()?, TOP_DOMAINS),
        model: ModelReport {
            server: ollama_info(&config.llm.base_url, &config.llm.model).await,
            summaries,
            embeddings,
            cost,
            local,
            pages_cut: stats.pages_cut,
            max_input_chars: config.llm.max_input_chars,
            fetch_secs_per_page: per(ctx.summary.fetch_secs, processed),
            model_secs_per_page: per(ctx.summary.model_secs, stats.model_pages),
            concurrency: config.run.concurrency,
        },
        tabkeeper: process_usage(),
        tags: TagCounts {
            in_use: topic.len(),
            new_this_run: db.tags_created_since(ctx.started_at)?,
            used_once: topic.iter().filter(|t| t.total == 1).count(),
            untagged_pages: db.untagged_pages()?,
            top,
        },
        review: ctx.review.clone(),
        warnings: Vec::new(),
        failed_pages: db
            .failed_pages()?
            .into_iter()
            .map(|(url, status, cause, error)| FailedPage {
                url,
                status,
                cause,
                error,
            })
            .collect(),
    };
    report.warnings = warnings(&report);
    Ok(report)
}

fn per(total: f64, n: usize) -> f64 {
    if n == 0 { 0.0 } else { total / n as f64 }
}

fn top_domains(urls: &[String], limit: usize) -> Vec<(String, usize)> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for url in urls {
        if let Some(host) = url::Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(str::to_string))
        {
            *counts
                .entry(host.trim_start_matches("www.").to_string())
                .or_default() += 1;
        }
    }
    let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    sorted.truncate(limit);
    sorted
}

/// Ollama's `/api/ps`: how big the loaded model is, how much of it is on the
/// GPU, and its context window. `None` for other servers or on any error.
async fn ollama_info(base_url: &str, model: &str) -> Option<ServerInfo> {
    let root = base_url.trim_end_matches('/').trim_end_matches("/v1");
    let port_is_ollama = url::Url::parse(root).ok().and_then(|u| u.port()) == Some(11434);
    if !is_local(base_url) && !port_is_ollama {
        return None;
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .ok()?;
    let body: serde_json::Value = client
        .get(format!("{root}/api/ps"))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    parse_ollama_ps(&body, model)
}

fn parse_ollama_ps(body: &serde_json::Value, model: &str) -> Option<ServerInfo> {
    let wanted = if model.contains(':') {
        model.to_string()
    } else {
        format!("{model}:latest")
    };
    let entry = body["models"]
        .as_array()?
        .iter()
        .find(|m| m["name"].as_str() == Some(wanted.as_str()))?;
    Some(ServerInfo {
        loaded_bytes: entry["size"].as_u64()?,
        gpu_bytes: entry["size_vram"].as_u64().unwrap_or(0),
        context_tokens: entry["context_length"].as_u64(),
    })
}

/// CPU time and peak memory of this process.
fn process_usage() -> ProcessUsage {
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: getrusage fills the struct it is given and only reads RUSAGE_SELF.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0 {
            // SAFETY: getrusage returned 0, so it initialized the struct.
            let usage = unsafe { usage.assume_init() };
            let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
            // ru_maxrss is in bytes on macOS and in kilobytes on Linux.
            let unit = if cfg!(target_os = "macos") { 1 } else { 1024 };
            return ProcessUsage {
                cpu_secs: Some(secs(usage.ru_utime) + secs(usage.ru_stime)),
                peak_memory_bytes: Some(usage.ru_maxrss as u64 * unit),
            };
        }
    }
    ProcessUsage::default()
}

/// Problems worth pointing out, each with what to do about it.
pub fn warnings(report: &Report) -> Vec<String> {
    let mut out = Vec::new();
    let usage = &report.model.summaries.usage;
    let answered = usage.requests.saturating_sub(usage.retries);
    let reported = answered.saturating_sub(usage.responses_without_usage);
    if let (Some(output_per_reply), Some(input_per_request)) = (
        usage.output_tokens.checked_div(reported),
        usage.input_tokens.checked_div(reported),
    ) {
        if output_per_reply > 1500 {
            out.push(format!(
                "Average {output_per_reply} output tokens per reply; the model may be thinking before it \
                 answers. On Ollama, set reasoning_effort = \"none\" in [llm.extra_body]."
            ));
        }
        if let Some(context) = report.model.server.as_ref().and_then(|s| s.context_tokens) {
            if input_per_request as f64 > 0.8 * context as f64 {
                out.push(format!(
                    "Prompts average {input_per_request} tokens but the model's context is {context}; the \
                     server may cut pages off without an error. Lower llm.max_input_chars or raise the context."
                ));
            }
        }
    }
    if let Some(server) = &report.model.server {
        let share = server.gpu_bytes as f64 / server.loaded_bytes.max(1) as f64;
        if share < 0.95 {
            out.push(format!(
                "Only {:.0}% of the model is on the GPU; a smaller model or a shorter context would be much faster.",
                share * 100.0
            ));
        }
    }
    let done = report.run.done;
    if done >= 10 && report.model.pages_cut * 10 >= done * 3 {
        out.push(format!(
            "{}% of pages were cut to fit llm.max_input_chars ({}); consider raising it if your model's context allows.",
            report.model.pages_cut * 100 / done,
            report.model.max_input_chars
        ));
    }
    if done >= 10 && usage.invalid_replies * 10 > done as u64 {
        out.push(format!(
            "{} replies were invalid JSON; try structured_output = \"json_schema\" or a larger model.",
            usage.invalid_replies
        ));
    }
    if let Some(embeddings) = report.model.embeddings.as_ref().filter(|e| e.errors > 0) {
        out.push(format!(
            "{} embedding requests failed; affected pages were shown the most used tags. The next run fills in \
             missing vectors.",
            embeddings.errors
        ));
    }
    out
}

/// The report as text, for the terminal and for `_report.md`.
pub fn text(report: &Report) -> String {
    let mut out = String::new();
    let started = report
        .started_at
        .replace('T', " ")
        .trim_end_matches('Z')
        .to_string();
    write!(
        out,
        "tabkeeper run {started} UTC, finished in {}",
        duration(report.duration_secs)
    )
    .unwrap();
    if let Some(reason) = &report.stopped {
        write!(out, "\nStopped early: {reason}").unwrap();
    }
    out.push_str("\n\nTabs\n");
    if let Some(import) = &report.import {
        row(&mut out, 2, "In the list", import.listed, "");
        row(
            &mut out,
            2,
            "Duplicates",
            import.duplicates,
            "repeated in the list",
        );
        row(
            &mut out,
            2,
            "Already known",
            import.known,
            "in the archive already",
        );
        let not_web = if import.not_web > 0 {
            format!(
                "({} by the filter · {} non-web)",
                num(import.filtered),
                num(import.not_web)
            )
        } else {
            String::new()
        };
        row(
            &mut out,
            2,
            "Filtered out",
            import.filtered + import.not_web,
            &not_web,
        );
        if !import.invalid.is_empty() {
            row(&mut out, 2, "Invalid", import.invalid.len(), "not URLs");
        }
    }
    let run = &report.run;
    row(&mut out, 2, "Processed", run.processed, "");
    row(&mut out, 4, "Done", run.done, "");
    row(
        &mut out,
        4,
        "Unreachable",
        run.unreachable,
        "stub notes tagged #status/unreachable",
    );
    causes_line(&mut out, &run.causes, true);
    row(
        &mut out,
        4,
        "Failed",
        run.failed,
        "stub notes tagged #status/failed",
    );
    causes_line(&mut out, &run.causes, false);
    if run.skipped_by_filter > 0 {
        row(
            &mut out,
            2,
            "Skipped",
            run.skipped_by_filter,
            "pending pages the filter excludes",
        );
    }
    row(
        &mut out,
        2,
        "Pending",
        run.pending,
        if run.pending > 0 {
            "left for the next run"
        } else {
            ""
        },
    );

    let a = &report.archive;
    writeln!(
        out,
        "\nArchive    {} notes: {} done · {} unreachable · {} failed",
        num(a.done + a.unreachable + a.failed),
        num(a.done),
        num(a.unreachable),
        num(a.failed)
    )
    .unwrap();
    if !report.languages.is_empty() {
        writeln!(out, "Languages  {}", list(&report.languages, 6)).unwrap();
    }
    if !report.domains.is_empty() {
        writeln!(out, "Domains    {}", list(&report.domains, TOP_DOMAINS)).unwrap();
    }

    let m = &report.model;
    let s = &m.summaries;
    out.push_str("\nModel\n");
    let options = if s.options.is_empty() {
        String::new()
    } else {
        format!(" · {}", s.options.join(", "))
    };
    writeln!(out, "  Summaries   {} at {}{options}", s.model, s.server).unwrap();
    if let Some(server) = &m.server {
        let share = server.gpu_bytes as f64 / server.loaded_bytes.max(1) as f64 * 100.0;
        let context = server.context_tokens.map_or(String::new(), |c| {
            format!(" · context {} tokens", num(c as usize))
        });
        writeln!(
            out,
            "              {:.1} GB loaded, {share:.0}% on GPU{context}",
            server.loaded_bytes as f64 / 1e9
        )
        .unwrap();
    }
    if let Some(e) = &m.embeddings {
        writeln!(
            out,
            "  Embeddings  {} · {} texts embedded ({} tokens) in {}",
            e.model,
            num(e.usage.texts_embedded as usize),
            tokens(e.usage.input_tokens),
            duration(e.usage.request_secs)
        )
        .unwrap();
    } else {
        out.push_str("  Embeddings  not used (the model was shown the most used tags)\n");
    }
    let u = &s.usage;
    writeln!(
        out,
        "  Requests    {} summary requests · {} retries · {} asked again after an invalid reply",
        num(u.requests as usize),
        num(u.retries as usize),
        num(u.invalid_replies as usize)
    )
    .unwrap();
    let answered = u
        .requests
        .saturating_sub(u.retries)
        .saturating_sub(u.responses_without_usage);
    if let (Some(input_each), Some(output_each)) = (
        u.input_tokens.checked_div(answered),
        u.output_tokens.checked_div(answered),
    ) {
        writeln!(
            out,
            "  Tokens      {} input · {} output · {} input / {} output per request",
            tokens(u.input_tokens),
            tokens(u.output_tokens),
            num(input_each as usize),
            num(output_each as usize)
        )
        .unwrap();
    } else if u.requests > 0 {
        out.push_str("  Tokens      not reported by the server\n");
    }
    if run.processed > 0 && report.duration_secs > 0.0 {
        // Over the whole run: with several pages at once, requests wait in the
        // server's queue, so time per request overstates the model's work.
        writeln!(
            out,
            "  Speed       {:.1} s per page{} · {:.0} output tokens/s",
            report.duration_secs / run.processed as f64,
            if m.concurrency > 0 {
                format!(" ({} at a time)", m.concurrency)
            } else {
                String::new()
            },
            u.output_tokens as f64 / report.duration_secs
        )
        .unwrap();
        writeln!(
            out,
            "  Time        a summary request took {:.1} s, including any wait in the server's queue · \
             a fetch {:.1} s",
            m.model_secs_per_page, m.fetch_secs_per_page
        )
        .unwrap();
    }
    let cost = match m.cost {
        Some(_) if m.local => "free (local)".to_string(),
        Some(cost) => format!("${cost:.2}"),
        None => "unknown: set price_input_per_mtok and price_output_per_mtok in [llm]".to_string(),
    };
    writeln!(out, "  Cost        {cost}").unwrap();
    if let Some(r) = report.review.as_ref().filter(|r| r.usage.requests > 0) {
        writeln!(
            out,
            "  Tag review  {} · {} requests · {} input / {} output tokens · {}",
            r.model,
            num(r.usage.requests as usize),
            tokens(r.usage.input_tokens),
            tokens(r.usage.output_tokens),
            duration(r.usage.request_secs)
        )
        .unwrap();
        if r.embed_usage.input_tokens > 0 {
            writeln!(
                out,
                "              plus {} embedding tokens",
                tokens(r.embed_usage.input_tokens)
            )
            .unwrap();
        }
    }
    if m.pages_cut > 0 {
        writeln!(
            out,
            "  Page limits {} pages cut to fit max_input_chars ({})",
            num(m.pages_cut),
            num(m.max_input_chars)
        )
        .unwrap();
    }
    if let (Some(cpu), Some(memory)) = (report.tabkeeper.cpu_secs, report.tabkeeper.peak_memory_bytes) {
        writeln!(
            out,
            "tabkeeper     CPU {} · peak memory {} MB",
            duration(cpu),
            memory / 1_000_000
        )
        .unwrap();
    }

    let t = &report.tags;
    writeln!(
        out,
        "\nTags       {} tags ({} new this run) · {} used once · {} untagged pages",
        num(t.in_use),
        num(t.new_this_run),
        num(t.used_once),
        num(t.untagged_pages)
    )
    .unwrap();
    let shared: Vec<(String, usize)> = t.top.iter().filter(|(_, n)| *n > 1).cloned().collect();
    if !shared.is_empty() {
        writeln!(out, "Most used  {}", list(&shared, TOP_TAGS)).unwrap();
    }
    if let Some(review) = &report.review {
        writeln!(out, "Review     {}", review.line()).unwrap();
    }
    if !report.warnings.is_empty() {
        out.push_str("\nWarnings\n");
        for warning in &report.warnings {
            writeln!(out, "  - {warning}").unwrap();
        }
    }
    out
}

/// The report for `_report.md`: the text, plus every failed and unreachable page.
pub fn markdown(report: &Report) -> String {
    let mut out = format!(
        "# tabkeeper report\n\n```\n{}```\n\nAll tags with page counts: [_tags.md](_tags.md)\n",
        text(report)
    );
    if !report.failed_pages.is_empty() {
        writeln!(
            out,
            "\n## Failed and unreachable pages ({})\n",
            report.failed_pages.len()
        )
        .unwrap();
        for page in &report.failed_pages {
            writeln!(
                out,
                "- {} ({}, {}): {}",
                page.url,
                page.status,
                page.cause.replace('_', " "),
                page.error
            )
            .unwrap();
        }
    }
    out
}

fn row(out: &mut String, indent: usize, label: &str, n: usize, note: &str) {
    let label = format!("{}{label}", " ".repeat(indent));
    writeln!(out, "{label:<17}{:>7}   {note}", num(n)).unwrap();
    // Trailing spaces from an empty note aren't wanted.
    let trimmed = out.trim_end_matches([' ', '\n']).len();
    out.truncate(trimmed);
    out.push('\n');
}

fn causes_line(out: &mut String, causes: &BTreeMap<String, usize>, unreachable: bool) {
    let parts: Vec<String> = causes
        .iter()
        .filter(|(kind, _)| is_unreachable_cause(kind) == unreachable)
        .map(|(kind, n)| format!("{} {}", kind.replace('_', " "), num(*n)))
        .collect();
    if !parts.is_empty() {
        writeln!(out, "      {}", parts.join(" · ")).unwrap();
    }
}

fn is_unreachable_cause(kind: &str) -> bool {
    matches!(
        kind,
        "not_found" | "blocked" | "http_error" | "connection" | "timeout"
    )
}

fn list(items: &[(String, usize)], limit: usize) -> String {
    let mut parts: Vec<String> = items
        .iter()
        .take(limit)
        .map(|(k, n)| format!("{k} {}", num(*n)))
        .collect();
    if items.len() > limit {
        parts.push("…".into());
    }
    parts.join(" · ")
}

/// 4812 → "4,812".
pub fn num(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// 9_800_000 → "9.8M".
pub fn tokens(n: u64) -> String {
    match n {
        0..1_000 => n.to_string(),
        1_000..1_000_000 => format!("{:.1}k", n as f64 / 1e3),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

/// Seconds as "41 s", "5m 3s" or "1h 42m".
pub fn duration(secs: f64) -> String {
    let s = secs.round() as u64;
    match s {
        0..60 => format!("{s} s"),
        60..3600 => format!("{}m {}s", s / 60, s % 60),
        _ => format!("{}h {}m", s / 3600, s % 3600 / 60),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn cloud_cost_is_unknown_without_summary_prices() {
        let db = Db::open_in_memory().unwrap();
        let mut config = Config::default();
        config.llm.base_url = "https://api.example.com/v1".into();
        config.embeddings.price_per_mtok = 0.02;
        let summary = RunSummary::default();
        let usage = StageUsage {
            input_tokens: 1_000_000,
            output_tokens: 100_000,
            ..Default::default()
        };
        let embeddings = StageUsage {
            input_tokens: 500_000,
            ..Default::default()
        };
        let ctx = |config| RunContext {
            started_at: "2026-01-01T00:00:00Z",
            started: std::time::Instant::now(),
            import: None,
            summary: &summary,
            summaries: usage.clone(),
            embeddings: Some((embeddings.clone(), 0)),
            review: None,
            config,
        };
        let mut priced = config.clone();
        priced.llm.price_input_per_mtok = 1.0;
        assert_eq!(build(&db, &ctx(&config)).await.unwrap().model.cost, None);
        let cost = build(&db, &ctx(&priced)).await.unwrap().model.cost.unwrap();
        assert!(
            (cost - 1.01).abs() < 1e-9,
            "$1 for summaries plus $0.01 for embeddings: {cost}"
        );
    }

    #[test]
    fn formats_numbers_tokens_and_durations() {
        assert_eq!(num(4812), "4,812");
        assert_eq!(num(1_234_567), "1,234,567");
        assert_eq!(num(12), "12");
        assert_eq!(tokens(950), "950");
        assert_eq!(tokens(9_800_000), "9.8M");
        assert_eq!(duration(41.2), "41 s");
        assert_eq!(duration(303.0), "5m 3s");
        assert_eq!(duration(6120.0), "1h 42m");
    }

    #[test]
    fn reads_ollama_ps() {
        let body = json!({"models": [{"name": "qwen3:4b", "size": 7_600_000_000u64, "size_vram": 4_560_000_000u64, "context_length": 32768}]});
        let info = parse_ollama_ps(&body, "qwen3:4b").unwrap();
        assert_eq!(
            (info.loaded_bytes, info.gpu_bytes, info.context_tokens),
            (7_600_000_000, 4_560_000_000, Some(32768))
        );
        assert!(parse_ollama_ps(&body, "llama3.1:8b").is_none());
        let latest = json!({"models": [{"name": "nomic-embed-text:latest", "size": 1, "size_vram": 1}]});
        assert!(parse_ollama_ps(&latest, "nomic-embed-text").is_some());
    }

    #[test]
    fn counts_domains_without_www() {
        let urls: Vec<String> = [
            "https://www.github.com/a",
            "https://github.com/b",
            "https://en.wikipedia.org/x",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(
            top_domains(&urls, 5),
            [("github.com".to_string(), 2), ("en.wikipedia.org".to_string(), 1)]
        );
    }

    fn sample() -> Report {
        let mut report = Report {
            started_at: "2026-09-30T14:02:00Z".into(),
            duration_secs: 6120.0,
            ..Default::default()
        };
        report.run = RunCounts {
            processed: 20,
            done: 17,
            unreachable: 2,
            failed: 1,
            causes: BTreeMap::from([("not_found".into(), 2), ("unsupported_type".into(), 1)]),
            ..Default::default()
        };
        report.model.summaries = Stage {
            model: "qwen3:4b".into(),
            server: "http://localhost:11434/v1".into(),
            usage: StageUsage {
                requests: 18,
                invalid_replies: 1,
                input_tokens: 54_000,
                output_tokens: 40_000,
                request_secs: 100.0,
                ..Default::default()
            },
            ..Default::default()
        };
        report.model.server = Some(ServerInfo {
            loaded_bytes: 100,
            gpu_bytes: 60,
            context_tokens: Some(4096),
        });
        report.model.cost = Some(0.0);
        report.model.local = true;
        report.model.pages_cut = 9;
        report.failed_pages = vec![FailedPage {
            url: "https://gone.example/".into(),
            status: "unreachable".into(),
            cause: "not_found".into(),
            error: "HTTP 404 Not Found".into(),
        }];
        report.warnings = warnings(&report);
        report
    }

    #[test]
    fn warns_about_detected_problems() {
        let warnings = sample().warnings;
        let joined = warnings.join("\n");
        assert!(joined.contains("2222 output tokens per reply"), "{joined}");
        assert!(joined.contains("Only 60% of the model is on the GPU"), "{joined}");
        assert!(joined.contains("52% of pages were cut"), "{joined}");
        assert!(
            !joined.contains("Prompts average"),
            "3,000 input tokens fit in 4,096: {joined}"
        );
        assert_eq!(warnings.len(), 3, "{joined}");
    }

    #[test]
    fn text_and_markdown() {
        let report = sample();
        let text = text(&report);
        assert!(
            text.starts_with("tabkeeper run 2026-09-30 14:02:00 UTC, finished in 1h 42m\n"),
            "{text}"
        );
        assert!(
            text.contains(
                "    Unreachable        2   stub notes tagged #status/unreachable\n      not found 2\n"
            ),
            "{text}"
        );
        assert!(text.contains("      unsupported type 1\n"), "{text}");
        assert!(text.contains("Cost        free (local)"), "{text}");
        assert!(
            !text.contains("gone.example"),
            "failed pages are only in the markdown"
        );
        let md = markdown(&report);
        assert!(
            md.contains("- https://gone.example/ (unreachable, not found): HTTP 404 Not Found"),
            "{md}"
        );
        // The JSON round-trips, so `report` can show the latest run later.
        let json = serde_json::to_string(&report).unwrap();
        assert_eq!(serde_json::from_str::<Report>(&json).unwrap().run.done, 17);
        // Reports saved by older versions, with fewer fields, still load.
        let old: Report =
            serde_json::from_str(r#"{"run": {"done": 3}, "model": {"summaries": {"model": "m"}}}"#).unwrap();
        assert_eq!((old.run.done, old.model.concurrency), (3, 0));
    }
}

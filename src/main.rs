mod config;
mod db;
mod embed;
mod estimate;
mod extract;
mod fetch;
mod filter;
mod import;
mod llm;
mod pipeline;
mod progress;
mod reconcile;
mod render;
mod report;
mod tags;
mod urls;
mod usage;

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

use crate::config::Config;
use crate::db::Db;
use crate::llm::LangMode;

/// Summarize and tag every open browser tab into Markdown notes.
#[derive(Parser)]
#[command(version)]
struct Cli {
    /// Output folder for notes and the database.
    #[arg(long, global = true, default_value = "tabkeeper-out")]
    out: PathBuf,

    /// Config file (default: the path shown by `tabkeeper config`).
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Import URLs from a text file (one per line), then summarize and tag them.
    Import {
        file: PathBuf,
        /// Language for titles and summaries: en, a language code such as de,
        /// or "original" for each page's own language.
        #[arg(long, default_value = "en")]
        lang: LangMode,
        /// Process at most this many pages in this run.
        #[arg(long)]
        limit: Option<usize>,
        /// Model to use, overriding the config file.
        #[arg(long)]
        model: Option<String>,
        /// Try pages that failed or were unreachable in earlier runs again.
        #[arg(long)]
        retry_failed: bool,
        /// Only process URLs matching this rule, e.g. domain:github.com. Can
        /// be repeated. Rules: domain:, glob:, regex:, prefix:.
        #[arg(long, value_name = "RULE")]
        allow: Vec<String>,
        /// Skip URLs matching this rule, e.g. domain:mail.google.com. Can be repeated.
        #[arg(long, value_name = "RULE")]
        deny: Vec<String>,
        /// Show what would be processed and an estimate of tokens, cost and
        /// time, without fetching, calling a model or saving anything.
        #[arg(long)]
        dry_run: bool,
        /// Stop starting new pages once this many tokens (input plus output)
        /// have been used.
        #[arg(long, value_name = "TOKENS")]
        max_tokens: Option<u64>,
        /// Stop starting new pages once this many US dollars have been spent,
        /// using the prices in the config.
        #[arg(long, value_name = "DOLLARS")]
        max_cost: Option<f64>,
        /// Print the final report as JSON instead of text.
        #[arg(long)]
        json: bool,
        /// Tabs processed at the same time, overriding run.concurrency.
        #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
        concurrency: Option<u32>,
        /// Maximum tags per page, overriding tags.max_per_page.
        #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
        max_tags: Option<u32>,
        /// Apply the tag review's changes at the end of the run without asking.
        #[arg(long)]
        yes: bool,
    },
    /// Review the tags: merge duplicates, split ambiguous tags. You approve
    /// the changes; `tabkeeper undo` reverts them.
    ReviseTags {
        /// Apply the changes without asking.
        #[arg(long)]
        yes: bool,
    },
    /// Undo the latest tag review that changed tags.
    Undo,
    /// Rewrite all notes, _tags.md and _index.md from the database.
    Render,
    /// Print the report of the latest run.
    Report {
        /// Print it as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show which config file is used, and an example config.
    Config,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<()> {
    let (mut config, config_path) = Config::load(cli.config.as_deref())?;
    match cli.command {
        Command::Import {
            file,
            lang,
            limit,
            model,
            retry_failed,
            allow,
            deny,
            dry_run,
            max_tokens,
            max_cost,
            json,
            concurrency,
            max_tags,
            yes,
        } => {
            if let Some(model) = model {
                config.llm.model = model;
            }
            if let Some(n) = concurrency {
                config.run.concurrency = n as usize;
            }
            if let Some(n) = max_tags {
                config.tags.max_per_page = n as usize;
            }
            let filter = filter::Filter::new(&config.filter, &allow, &deny)?;
            let text =
                std::fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?;
            if dry_run {
                return estimate::dry_run(&cli.out, &text, &filter, &config, retry_failed, limit);
            }
            if max_cost.is_some() && !usage::is_local(&config.llm.base_url) && !config.llm.prices().is_set() {
                anyhow::bail!(
                    "--max-cost needs prices: set price_input_per_mtok and price_output_per_mtok in [llm]"
                );
            }

            // With --json, stdout is only for the report; everything else goes to stderr.
            let say = |text: &str| {
                if json {
                    eprintln!("{text}")
                } else {
                    println!("{text}")
                }
            };
            let started_at = db::now();
            let started = std::time::Instant::now();
            let mut db = open_db(&cli.out)?;
            let stats = import::import_url_list(&db, &text, &filter)?;
            say(&format!(
                "Imported {} new URLs ({} already known, {} duplicates, {} skipped by the filter, {} non-web).",
                stats.added, stats.known, stats.duplicates, stats.filtered, stats.not_web
            ));
            for bad in &stats.invalid {
                say(&format!("Skipped invalid URL: {bad}"));
            }
            if retry_failed {
                say(&format!(
                    "Retrying {} failed and unreachable pages.",
                    db.retry_failed()?
                ));
            }

            let fetcher = fetch::HttpFetcher::new(&config.fetch)?;
            let llm = llm::OpenAiCompatible::new(&config.llm)?;
            say(&format!(
                "Using model {} at {}.",
                config.llm.model, config.llm.base_url
            ));

            // Embeddings are optional: any problem setting them up is a warning,
            // and the model is shown the most used tags instead.
            let unavailable = |err: anyhow::Error| {
                say(&format!(
                    "Warning: embeddings are unavailable ({err:#}).\n\
                     The model will be shown the most used tags instead of the most relevant ones. \
                     If the embedding model is missing on Ollama, run: ollama pull {}",
                    config.embeddings.model
                ))
            };
            let embedder = if config.embeddings.enabled {
                embed::OpenAiEmbedder::new(&config.embeddings, &config.llm)
                    .map_err(unavailable)
                    .ok()
            } else {
                None
            };
            let mut embeddings = None;
            if let Some(embedder) = &embedder {
                match pipeline::Embeddings::prepare(&db, embedder, config.embeddings.page_chars).await {
                    Ok(e) => {
                        say(&format!(
                            "Using embedding model {} to pick relevant tags.",
                            config.embeddings.model
                        ));
                        embeddings = Some(e);
                    }
                    Err(err) => unavailable(err),
                }
            }

            let budget = pipeline::Budget { max_tokens, max_cost };
            // An embeddings setup that failed partway still spent tokens: count
            // them toward the budget, since the run only tracks working embeddings.
            let failed_setup = match (&embedder, &embeddings) {
                (Some(embedder), None) => embed::Embedder::usage(embedder),
                _ => usage::StageUsage::default(),
            };
            let options = pipeline::RunOptions {
                config: &config,
                lang: &lang,
                filter: &filter,
                limit,
                budget,
                progress_to_stderr: json,
                spent_elsewhere: (
                    failed_setup.total_tokens(),
                    failed_setup.input_tokens as f64 * config.embeddings.price_per_mtok / 1e6,
                ),
            };
            let result =
                pipeline::process_pending(&mut db, &fetcher, &llm, embeddings.as_mut(), &options).await;

            // Render and report whatever was finished, even if the run stopped early.
            render::render_all(&db, &cli.out, config.notes.title)?;
            let mut summary = result?;
            // A run stopped by an error skips the review, which would make
            // more model requests; so does a run with a budget, which the
            // review could take it over.
            let budgeted = max_tokens.is_some() || max_cost.is_some();
            if config.reconcile.at_end_of_run && summary.stop.is_none() && budgeted {
                say(
                    "The tag review was skipped because of --max-tokens/--max-cost; run `tabkeeper revise-tags` for it.",
                );
            }
            let review = if config.reconcile.at_end_of_run && summary.stop.is_none() && !budgeted {
                let review = review_tags(&mut db, &config, yes, &say).await;
                render::render_all(&db, &cli.out, config.notes.title)?;
                Some(review)
            } else {
                None
            };
            let embed_usage = match (&embedder, &embeddings) {
                (Some(embedder), Some(e)) => Some((embed::Embedder::usage(embedder), e.errors)),
                // The setup failed: report what it used, and count the failure.
                (Some(_), None) if failed_setup.requests > 0 => Some((failed_setup.clone(), 1)),
                _ => None,
            };
            let ctx = report::RunContext {
                started_at: &started_at,
                started,
                import: Some(&stats),
                summary: &summary,
                summaries: llm::Llm::usage(&llm),
                embeddings: embed_usage,
                review,
                config: &config,
            };
            let report = report::build(&db, &ctx).await?;
            db.save_run(&started_at, &report.finished_at, &serde_json::to_string(&report)?)?;
            let report_path = cli.out.join("_report.md");
            std::fs::write(&report_path, report::markdown(&report))?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print!("\n{}", report::text(&report));
                println!(
                    "\nNotes, _index.md and the full report (with every failed page) are in {}.",
                    cli.out.display()
                );
            }
            if let Some(pipeline::Stop::Error(err)) = summary.stop.take() {
                return Err(err);
            }
        }
        Command::Render => finish(&open_db(&cli.out)?, &cli.out, &config)?,
        Command::ReviseTags { yes } => {
            let mut db = open_db(&cli.out)?;
            let review = review_tags(&mut db, &config, yes, &|text: &str| println!("{text}")).await;
            finish(&db, &cli.out, &config)?;
            if let Some(error) = &review.error {
                bail!("the tag review failed: {error}");
            }
            println!("Tag review: {}.", review.line());
            if let Some(revision) = review.revision {
                println!("To revert it: tabkeeper undo (revision {revision}).");
            }
        }
        Command::Undo => {
            let mut db = open_db(&cli.out)?;
            match db.undo_last_revision()? {
                Some((revision, changes)) => {
                    finish(&db, &cli.out, &config)?;
                    println!("Undid tag review revision {revision} ({changes} changes).");
                }
                None => println!("No tag review to undo."),
            }
        }
        Command::Report { json } => {
            let db = open_db(&cli.out)?;
            match db.latest_run_report()? {
                Some(saved) if json => println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::from_str::<serde_json::Value>(&saved)?)?
                ),
                Some(saved) => print!("{}", report::text(&serde_json::from_str(&saved)?)),
                None => {
                    println!("No runs recorded yet. Tags so far:\n");
                    print!(
                        "{}",
                        render::report(&tags::count_tree(&db.tags()?, &db.tag_links()?))
                    );
                }
            }
        }
        Command::Config => {
            match (&config_path, config::default_path()) {
                (Some(p), _) => println!("Using config file {}.", p.display()),
                (None, Some(p)) => println!(
                    "No config file; using defaults. Create {} to change them.",
                    p.display()
                ),
                (None, None) => println!("No config file; using defaults."),
            }
            print!("\nExample config:\n\n{}", config::EXAMPLE);
        }
    }
    Ok(())
}

/// Runs the tag review and asks which changes to apply. A failure is reported,
/// not returned: the run's pages are saved either way.
async fn review_tags(db: &mut Db, config: &Config, yes: bool, say: &dyn Fn(&str)) -> report::ReviewReport {
    let llm_config = config.reconcile.llm_config(&config.llm);
    let mut outcome = report::ReviewReport {
        model: llm_config.model.clone(),
        ..Default::default()
    };
    let reviewer = match llm::OpenAiCompatible::new(&llm_config) {
        Ok(reviewer) => reviewer,
        Err(err) => {
            outcome.error = Some(format!("{err:#}"));
            return outcome;
        }
    };
    // Without embeddings, only plural/singular pairs are checked. Vectors
    // missing from earlier runs are filled in first: splits need page vectors.
    let embedder = if config.embeddings.enabled {
        embed::OpenAiEmbedder::new(&config.embeddings, &config.llm).ok()
    } else {
        None
    };
    let mut usable = embedder.as_ref();
    if let Some(e) = usable {
        if let Err(err) = pipeline::Embeddings::prepare(db, e, config.embeddings.page_chars).await {
            say(&format!(
                "Warning: embeddings are unavailable ({err:#}); the review only checks plural/singular pairs."
            ));
            usable = None;
        }
    }
    let result = reconcile::review(db, &reviewer, usable, &config.reconcile, say).await;
    outcome.usage = llm::Reviewer::usage(&reviewer);
    outcome.embed_usage = embedder.as_ref().map(embed::Embedder::usage).unwrap_or_default();
    let review = match result {
        Ok(review) => review,
        Err(err) => {
            outcome.error = Some(format!("{err:#}"));
            return outcome;
        }
    };
    outcome.merge_candidates = review.merge_candidates;
    outcome.split_candidates = review.split_candidates;
    outcome.proposed = review.proposed.len();
    if review.proposed.is_empty() {
        return outcome;
    }

    say("\nThe tag review proposes these changes:");
    for (i, p) in review.proposed.iter().enumerate() {
        say(&format!("  {:>2}. {}", i + 1, p.description));
    }
    let chosen: Vec<bool> = if yes {
        vec![true; review.proposed.len()]
    } else if !std::io::stdin().is_terminal() {
        outcome.not_asked = review.proposed.len();
        return outcome;
    } else {
        match ask_which(review.proposed.len()) {
            Some(chosen) => chosen,
            // No answer: nothing is applied, and nothing counts as declined.
            None => {
                outcome.not_asked = review.proposed.len();
                return outcome;
            }
        }
    };

    let mut apply = Vec::new();
    let mut keys = Vec::new();
    for (p, keep) in review.proposed.iter().zip(&chosen) {
        if *keep {
            match &p.change {
                db::TagChange::Merge { .. } => outcome.merges += 1,
                db::TagChange::Rename { .. } => outcome.renames += 1,
                db::TagChange::Split { .. } => outcome.splits += 1,
            }
            apply.push(p.change.clone());
            keys.push(p.decision_keys.clone());
        } else {
            outcome.declined += 1;
            for key in &p.decision_keys {
                if let Err(err) = db.set_tag_decision(key, "declined", "user") {
                    say(&format!("Warning: couldn't remember a declined change: {err:#}"));
                }
            }
        }
    }
    if !apply.is_empty() {
        match db.apply_reviewed_changes(&apply, &keys) {
            Ok(applied) => {
                outcome.revision = applied.revision;
                for (i, reason) in applied.failed {
                    match &apply[i] {
                        db::TagChange::Merge { .. } => outcome.merges -= 1,
                        db::TagChange::Rename { .. } => outcome.renames -= 1,
                        db::TagChange::Split { .. } => outcome.splits -= 1,
                    }
                    let description = review
                        .proposed
                        .iter()
                        .find(|p| p.change == apply[i])
                        .map_or("a change", |p| p.description.as_str());
                    outcome.failed.push(format!(
                        "{}: {reason}",
                        description.split_whitespace().collect::<Vec<_>>().join(" ")
                    ));
                }
            }
            Err(err) => {
                outcome.error = Some(format!("applying the changes: {err:#}"));
                outcome.merges = 0;
                outcome.renames = 0;
                outcome.splits = 0;
            }
        }
    }
    outcome
}

/// Asks which of `n` numbered changes to apply: all (the default), none, or
/// all except the listed numbers. Asked on stderr, so --json output stays
/// clean. `None` if there was no answer.
fn ask_which(n: usize) -> Option<Vec<bool>> {
    loop {
        eprint!("Apply them? [Y]es, [n]o, or the numbers to skip (e.g. 2,5): ");
        let mut answer = String::new();
        // End of input (Ctrl-D) or an error: no answer.
        if !matches!(std::io::stdin().read_line(&mut answer), Ok(read) if read > 0) {
            eprintln!();
            return None;
        }
        match parse_choice(&answer, n) {
            Some(chosen) => return Some(chosen),
            None => eprintln!("Please answer y, n, or numbers between 1 and {n}."),
        }
    }
}

fn parse_choice(answer: &str, n: usize) -> Option<Vec<bool>> {
    match answer.trim().to_lowercase().as_str() {
        "" | "y" | "yes" | "a" | "all" => Some(vec![true; n]),
        "n" | "no" | "none" => Some(vec![false; n]),
        numbers => {
            let mut chosen = vec![true; n];
            for part in numbers
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|p| !p.is_empty())
            {
                let i: usize = part.parse().ok()?;
                *chosen.get_mut(i.checked_sub(1)?)? = false;
            }
            Some(chosen)
        }
    }
}

fn open_db(out: &Path) -> Result<Db> {
    std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    Db::open(&out.join("tabkeeper.db"))
}

/// Renders the notes and prints what's in the archive.
fn finish(db: &Db, out: &Path, config: &Config) -> Result<()> {
    let rendered = render::render_all(db, out, config.notes.title)?;
    let counts = db.status_counts()?;
    let causes = db.failure_kinds()?;
    if !causes.is_empty() {
        let list: Vec<String> = causes
            .iter()
            .map(|(kind, n)| format!("{} {n}", kind.replace('_', " ")))
            .collect();
        println!("\nUnreachable or failed pages by cause: {}.", list.join(" · "));
    }
    println!(
        "\n{} notes in {}. Pages: {} done, {} unreachable, {} failed, {} pending.",
        rendered.notes,
        out.display(),
        counts.done,
        counts.unreachable,
        counts.failed,
        counts.pending
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_choice;

    #[test]
    fn approval_answers() {
        assert_eq!(parse_choice("\n", 3), Some(vec![true, true, true]));
        assert_eq!(parse_choice("n", 3), Some(vec![false, false, false]));
        assert_eq!(parse_choice("2, 3", 3), Some(vec![true, false, false]));
        assert_eq!(parse_choice("4", 3), None);
        assert_eq!(parse_choice("0", 3), None);
        assert_eq!(parse_choice("maybe", 3), None);
    }
}

mod config;
mod db;
mod embed;
mod extract;
mod fetch;
mod filter;
mod import;
mod llm;
mod pipeline;
mod progress;
mod render;
mod tags;
mod urls;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
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
    },
    /// Rewrite all notes, _tags.md and _index.md from the database.
    Render,
    /// Print the tag tree with page counts.
    Report,
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
        } => {
            if let Some(model) = model {
                config.llm.model = model;
            }
            let filter = filter::Filter::new(&config.filter, &allow, &deny)?;
            let text =
                std::fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?;
            let mut db = open_db(&cli.out)?;
            let stats = import::import_url_list(&db, &text, &filter)?;
            println!(
                "Imported {} new URLs ({} already known, {} skipped by the filter, {} non-web skipped).",
                stats.added, stats.known, stats.filtered, stats.not_web
            );
            for bad in &stats.invalid {
                println!("Skipped invalid URL: {bad}");
            }
            if retry_failed {
                println!("Retrying {} failed and unreachable pages.", db.retry_failed()?);
            }

            let fetcher = fetch::HttpFetcher::new(&config.fetch)?;
            let llm = llm::OpenAiCompatible::new(&config.llm)?;
            println!("Using model {} at {}.", config.llm.model, config.llm.base_url);

            // Embeddings are optional: any problem setting them up is a warning,
            // and the model is shown the most used tags instead.
            let unavailable = |err: anyhow::Error| {
                println!(
                    "Warning: embeddings are unavailable ({err:#}).\n\
                     The model will be shown the most used tags instead of the most relevant ones. \
                     If the embedding model is missing on Ollama, run: ollama pull {}",
                    config.embeddings.model
                )
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
                        println!(
                            "Using embedding model {} to pick relevant tags.",
                            config.embeddings.model
                        );
                        embeddings = Some(e);
                    }
                    Err(err) => unavailable(err),
                }
            }

            let options = pipeline::RunOptions {
                config: &config,
                lang: &lang,
                filter: &filter,
                limit,
            };
            let result =
                pipeline::process_pending(&mut db, &fetcher, &llm, embeddings.as_mut(), &options).await;
            // Render whatever was finished, even if the run stopped early.
            finish(&db, &cli.out)?;
            if let Some(errors) = embeddings.as_ref().map(|e| e.errors).filter(|n| *n > 0) {
                println!("{errors} embedding requests failed; the next run fills in the missing vectors.");
            }
            let stats = result?;
            println!(
                "This run: {} done, {} unreachable, {} failed (both get stub notes).",
                stats.done, stats.unreachable, stats.failed
            );
            if stats.filtered > 0 {
                println!(
                    "{} pending pages were skipped by the filter and stay pending.",
                    stats.filtered
                );
            }
        }
        Command::Render => finish(&open_db(&cli.out)?, &cli.out)?,
        Command::Report => {
            let db = open_db(&cli.out)?;
            print!(
                "{}",
                render::report(&tags::count_tree(&db.tags()?, &db.tag_links()?))
            );
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

fn open_db(out: &Path) -> Result<Db> {
    std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    Db::open(&out.join("tabkeeper.db"))
}

/// Renders the notes and prints the tag report and page status.
fn finish(db: &Db, out: &Path) -> Result<()> {
    let rendered = render::render_all(db, out)?;
    print!(
        "\n{}",
        render::report(&tags::count_tree(&db.tags()?, &db.tag_links()?))
    );
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

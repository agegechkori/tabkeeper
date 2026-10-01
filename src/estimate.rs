use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;

use crate::config::Config;
use crate::db::Db;
use crate::filter::Filter;
use crate::import::parse_url_list;
use crate::report::{Report, duration, num, tokens};
use crate::urls::{Rejected, normalize};
use crate::usage::is_local;

/// What a run would process, for `--dry-run`.
#[derive(Debug, Default, PartialEq)]
pub struct Plan {
    pub listed: usize,
    pub duplicates: usize,
    pub filtered: usize,
    pub not_web: usize,
    pub invalid: usize,
    /// In the archive and not to be processed again.
    pub already_done: usize,
    pub new: usize,
    /// Pending in the archive (from this list or earlier ones), or failed and
    /// retried with --retry-failed.
    pub from_archive: usize,
}

impl Plan {
    pub fn to_process(&self, limit: Option<usize>) -> usize {
        let all = self.new + self.from_archive;
        limit.map_or(all, |n| all.min(n))
    }
}

/// Works out which URLs a run would process, without changing anything.
pub fn plan(db: Option<&Db>, text: &str, filter: &Filter, retry_failed: bool) -> Result<Plan> {
    let mut plan = Plan::default();
    let mut seen = HashSet::new();
    for (raw, _) in parse_url_list(text) {
        plan.listed += 1;
        match normalize(raw) {
            Ok(url) if !seen.insert(url.clone()) => plan.duplicates += 1,
            Ok(url) if !filter.allows(&url) => plan.filtered += 1,
            Ok(url) => match db
                .map(|db| db.page_status(&url))
                .transpose()?
                .flatten()
                .as_deref()
            {
                None => plan.new += 1,
                // Counted below with the rest of the archive's pending pages.
                Some("pending") => {}
                Some("failed" | "unreachable") if retry_failed => {}
                Some(_) => plan.already_done += 1,
            },
            Err(Rejected::NotWeb(_)) => plan.not_web += 1,
            Err(Rejected::Invalid) => plan.invalid += 1,
        }
    }
    if let Some(db) = db {
        let mut statuses = vec!["pending"];
        if retry_failed {
            statuses.extend(["failed", "unreachable"]);
        }
        plan.from_archive = db
            .urls_with_status(&statuses)?
            .iter()
            .filter(|url| filter.allows(url))
            .count();
    }
    Ok(plan)
}

/// Tokens and time per page: from the latest run if it reported them,
/// otherwise a rough guess from the config.
struct PerPage {
    input_tokens: f64,
    output_tokens: f64,
    secs: Option<f64>,
    from_last_run: bool,
}

fn per_page(last: Option<&Report>, config: &Config) -> PerPage {
    if let Some(report) = last {
        let usage = &report.model.summaries.usage;
        let answered = usage
            .requests
            .saturating_sub(usage.retries)
            .saturating_sub(usage.responses_without_usage);
        let processed = report.run.processed;
        if answered > 0 && processed > 0 {
            return PerPage {
                input_tokens: usage.input_tokens as f64 / answered as f64,
                output_tokens: usage.output_tokens as f64 / answered as f64,
                secs: Some(report.duration_secs / processed as f64),
                from_last_run: true,
            };
        }
    }
    // About 4 characters per token: the instructions, the tag list, and a page
    // filling about half of max_input_chars.
    let chars =
        2_000.0 + config.tags.vocabulary_limit as f64 * 60.0 + config.llm.max_input_chars as f64 * 0.5;
    PerPage {
        input_tokens: chars / 4.0,
        output_tokens: 250.0,
        secs: None,
        from_last_run: false,
    }
}

pub fn dry_run(
    out: &Path,
    text: &str,
    filter: &Filter,
    config: &Config,
    retry_failed: bool,
    limit: Option<usize>,
) -> Result<()> {
    let db_path = out.join("tabkeeper.db");
    let db = if db_path.exists() {
        Some(Db::open(&db_path)?)
    } else {
        None
    };
    let plan = plan(db.as_ref(), text, filter, retry_failed)?;
    let last: Option<Report> = match &db {
        Some(db) => db
            .latest_run_report()?
            .and_then(|json| serde_json::from_str(&json).ok()),
        None => None,
    };
    let pages = plan.to_process(limit);

    println!("Dry run: nothing is fetched, sent to a model or saved.\n");
    println!(
        "URLs in the list   {:>7}   (duplicates {} · filtered out {} · non-web {} · invalid {})",
        num(plan.listed),
        num(plan.duplicates),
        num(plan.filtered),
        num(plan.not_web),
        num(plan.invalid)
    );
    println!(
        "Already done       {:>7}   in the archive, skipped",
        num(plan.already_done)
    );
    let limited = if pages < plan.new + plan.from_archive {
        " (--limit)"
    } else {
        ""
    };
    println!(
        "To process         {:>7}   {} new, {} from the archive{limited}",
        num(pages),
        num(plan.new),
        num(plan.from_archive)
    );
    if pages == 0 {
        return Ok(());
    }

    let each = per_page(last.as_ref(), config);
    let input = (each.input_tokens * pages as f64) as u64;
    let output = (each.output_tokens * pages as f64) as u64;
    let basis = if each.from_last_run {
        "averages from your last run"
    } else {
        "a rough guess; a first run gives better numbers"
    };
    println!("\nEstimate for {} pages ({basis}):", num(pages));
    println!("  Tokens   ~{} input · ~{} output", tokens(input), tokens(output));
    let prices = config.llm.prices();
    let cost = if is_local(&config.llm.base_url) {
        "free (local)".to_string()
    } else if prices.is_set() {
        format!("~${:.2}", prices.cost(input, output))
    } else {
        "unknown: set price_input_per_mtok and price_output_per_mtok in [llm]".to_string()
    };
    println!("  Cost     {cost}");
    if let Some(secs) = each.secs {
        println!(
            "  Time     ~{} at the last run's speed",
            duration(secs * pages as f64)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::PageResult;

    #[test]
    fn plans_without_changing_anything() {
        let db = Db::open_in_memory().unwrap();
        for url in [
            "https://done.com/",
            "https://pending.com/",
            "https://failed.com/",
            "https://other-pending.com/",
        ] {
            db.add_page(url, url, None, "import").unwrap();
        }
        let mut db = db;
        let ids: Vec<i64> = db.pending_pages().unwrap().iter().map(|p| p.id).collect();
        db.save_result(
            ids[0],
            &PageResult {
                title: "T",
                summary: "S",
                lang: None,
                tags: &[],
            },
        )
        .unwrap();
        db.save_failed(
            ids[2],
            &PageResult {
                title: "T",
                summary: "S",
                lang: None,
                tags: &[],
            },
            "timeout",
            "x",
        )
        .unwrap();

        let list = "https://done.com/\nhttps://pending.com/\nhttps://failed.com/\nhttps://new.com/\nhttps://new.com/#dup\nchrome://x\n";
        let filter = Filter::default();
        let p = plan(Some(&db), list, &filter, false).unwrap();
        assert_eq!(
            p,
            Plan {
                listed: 6,
                duplicates: 1,
                not_web: 1,
                already_done: 2,
                new: 1,
                from_archive: 2,
                ..Plan::default()
            }
        );
        assert_eq!(p.to_process(None), 3);
        assert_eq!(p.to_process(Some(2)), 2);

        let p = plan(Some(&db), list, &filter, true).unwrap();
        assert_eq!(
            (p.already_done, p.from_archive),
            (1, 3),
            "--retry-failed brings the failed page back"
        );
        assert_eq!(db.pending_pages().unwrap().len(), 2, "nothing was changed");

        let p = plan(None, list, &filter, false).unwrap();
        assert_eq!(
            (p.new, p.from_archive),
            (4, 0),
            "without an archive everything is new"
        );
    }

    #[test]
    fn estimates_from_the_last_run_when_there_is_one() {
        let config = Config::default();
        let guess = per_page(None, &config);
        assert!(!guess.from_last_run);
        let mut last = Report::default();
        last.run.processed = 10;
        last.duration_secs = 100.0;
        last.model.summaries.usage.requests = 11;
        last.model.summaries.usage.retries = 1;
        last.model.summaries.usage.input_tokens = 30_000;
        last.model.summaries.usage.output_tokens = 2_500;
        let each = per_page(Some(&last), &config);
        assert_eq!(
            (each.input_tokens, each.output_tokens, each.secs),
            (3_000.0, 250.0, Some(10.0))
        );
    }
}

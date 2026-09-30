use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};

use crate::config::{Config, TagConfig};
use crate::db::{Db, PageResult, PendingPage};
use crate::extract::extract;
use crate::fetch::Fetcher;
use crate::llm::{LangMode, Llm, PageRejected, PageSummary, SummaryRequest};
use crate::tags::{TagPath, count_tree, vocabulary};

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub done: usize,
    pub failed: usize,
}

enum PageError {
    /// Only this page failed; record it and move on.
    Page(anyhow::Error),
    /// Every page would fail (e.g. the LLM server is down); stop the run.
    Fatal(anyhow::Error),
}

impl From<anyhow::Error> for PageError {
    fn from(err: anyhow::Error) -> Self {
        Self::Fatal(err)
    }
}

/// Processes pending pages one at a time. Each page is saved as soon as it
/// is done, so an interrupted run continues where it stopped.
pub async fn process_pending<F: Fetcher, L: Llm>(
    db: &mut Db,
    fetcher: &F,
    llm: &L,
    config: &Config,
    lang: &LangMode,
    limit: Option<usize>,
) -> Result<Stats> {
    let mut pending = db.pending_pages()?;
    if let Some(n) = limit {
        pending.truncate(n);
    }
    let total = pending.len();
    let mut stats = Stats::default();
    for (i, page) in pending.iter().enumerate() {
        let progress = format!("[{}/{}]", i + 1, total);
        match process_one(db, fetcher, llm, config, lang, page).await {
            Ok(title) => {
                stats.done += 1;
                println!("{progress} done    {} -> {title}", page.url);
            }
            Err(PageError::Page(err)) => {
                stats.failed += 1;
                db.mark_failed(page.id, &format!("{err:#}"))?;
                println!("{progress} failed  {}: {err:#}", page.url);
            }
            Err(PageError::Fatal(err)) => {
                return Err(err.context(format!(
                    "stopped at {} ({} done, {} failed so far; run again to continue)",
                    page.url, stats.done, stats.failed
                )));
            }
        }
    }
    Ok(stats)
}

async fn process_one<F: Fetcher, L: Llm>(
    db: &mut Db,
    fetcher: &F,
    llm: &L,
    config: &Config,
    lang: &LangMode,
    page: &PendingPage,
) -> Result<String, PageError> {
    let fetched = fetcher.fetch(&page.url).await.map_err(PageError::Page)?;
    let extracted = extract(&fetched.html, &fetched.final_url, config.llm.max_input_chars);
    let title = [Some(extracted.title.as_str()), page.browser_title.as_deref()]
        .into_iter()
        .flatten()
        .find(|t| !t.trim().is_empty())
        .unwrap_or("");
    let detected_lang = whatlang::detect(&extracted.text)
        .filter(|info| info.is_reliable())
        .map(|info| info.lang().eng_name());

    let counted = count_tree(&db.tags()?, &db.tag_links()?);
    let vocab = vocabulary(&counted, config.tags.vocabulary_limit);
    let request = SummaryRequest {
        url: &page.url,
        title,
        text: &extracted.text,
        lang,
        detected_lang,
        vocabulary: &vocab,
        tags: &config.tags,
    };
    let summary = llm.summarize(&request).await.map_err(|err| {
        if err.downcast_ref::<PageRejected>().is_some() {
            PageError::Page(err)
        } else {
            PageError::Fatal(err.context("LLM request failed"))
        }
    })?;

    let tags = resolve_tags(db, &summary, &config.tags)?;
    let language = summary.language.trim().to_ascii_lowercase();
    db.save_result(
        page.id,
        &PageResult {
            title: summary.title.trim(),
            summary: summary.summary.trim(),
            lang: (!language.is_empty()).then_some(language.as_str()),
            tags: &tags,
        },
    )
    .context("saving result")?;
    Ok(summary.title.trim().to_string())
}

/// Normalizes the model's tags, enforces the per-page limits, and creates
/// new tags (with their descriptions) as needed.
fn resolve_tags(db: &Db, summary: &PageSummary, limits: &TagConfig) -> Result<Vec<(String, i64)>> {
    let descriptions: HashMap<TagPath, &str> = summary
        .new_tags
        .iter()
        .filter_map(|t| {
            let path = TagPath::parse(&t.path, limits.max_depth)?;
            let desc = t.description.trim();
            (!desc.is_empty()).then_some((path, desc))
        })
        .collect();

    let mut seen = HashSet::new();
    let candidates: Vec<(&String, TagPath)> = summary
        .tags
        .iter()
        .filter_map(|raw| Some((raw, TagPath::parse(raw, limits.max_depth)?)))
        .filter(|(_, path)| !path.is_reserved() && seen.insert(path.clone()))
        .collect();
    // A tag is implied by its descendants, so drop it when one is present:
    // programming-languages + programming-languages/rust keeps only the latter.
    let has_descendant = |path: &TagPath| {
        candidates.iter().any(|(_, other)| {
            other.segments().len() > path.segments().len() && other.segments().starts_with(path.segments())
        })
    };

    let mut resolved = Vec::new();
    let mut new_count = 0;
    for (raw, path) in candidates.iter().filter(|(_, path)| !has_descendant(path)) {
        if resolved.len() >= limits.max_per_page {
            break;
        }
        if db.tag_id(path)?.is_none() {
            // Over the new-tag limit: skip, unless the page would end up untagged.
            if new_count >= limits.max_new_per_page && !resolved.is_empty() {
                continue;
            }
            new_count += 1;
        }
        let id = db.ensure_tag(path, descriptions.get(path).copied())?;
        resolved.push(((*raw).clone(), id));
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use anyhow::bail;

    use super::*;
    use crate::fetch::FetchedPage;
    use crate::llm::NewTag;

    struct FakeFetcher;

    impl Fetcher for FakeFetcher {
        async fn fetch(&self, url: &str) -> Result<FetchedPage> {
            if url.contains("broken") {
                bail!("HTTP 404 Not Found");
            }
            Ok(FetchedPage {
                final_url: url.to_string(),
                html: format!(
                    "<html><head><title>Page {url}</title></head><body><article><p>{}</p></article></body></html>",
                    "Rust is a systems programming language focused on safety and speed. ".repeat(10)
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
                Err("reject") => Err(PageRejected("too long".into()).into()),
                Err(other) => bail!("{other}"),
            }
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
                .map(|(p, d)| NewTag {
                    path: p.to_string(),
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
            &Config::default(),
            &LangMode::English,
            None,
        )
        .await
    }

    #[tokio::test]
    async fn processes_pages_and_grows_vocabulary() {
        let mut db = db_with(&["https://a.com/", "https://broken.com/", "https://b.com/"]);
        let llm = FakeLlm::new(vec![
            Ok(summary(
                "Rust intro",
                &["Technology/Programming Languages/Rust"],
                &[("technology/programming-languages/rust", "The Rust language")],
            )),
            Ok(summary(
                "More Rust",
                &["technology/programming-languages/rust", "status/unreachable"],
                &[],
            )),
        ]);
        let stats = run(&mut db, &llm).await.unwrap();
        assert_eq!(stats, Stats { done: 2, failed: 1 });

        // The second page was shown the tag the first page created.
        let seen = llm.seen_vocab.borrow();
        assert_eq!(seen[0], "");
        assert!(
            seen[1].contains("technology/programming-languages/rust (1) - The Rust language"),
            "{}",
            seen[1]
        );

        // The reserved status/ tag from the model was dropped.
        let tags = db.tags().unwrap();
        assert!(tags.iter().all(|t| !t.path.starts_with("status")));
        let pages = db.done_pages().unwrap();
        assert_eq!(pages[0].lang.as_deref(), Some("en"));
        assert_eq!(db.status_counts().unwrap().failed, 1);
    }

    #[tokio::test]
    async fn enforces_new_tag_limit() {
        let mut db = db_with(&["https://a.com/"]);
        let llm = FakeLlm::new(vec![Ok(summary("T", &["a", "b", "c", "d"], &[]))]);
        run(&mut db, &llm).await.unwrap();
        let paths: Vec<String> = db.tags().unwrap().into_iter().map(|t| t.path).collect();
        assert_eq!(paths, ["a", "b"]);
    }

    #[tokio::test]
    async fn drops_tags_implied_by_a_descendant() {
        let mut db = db_with(&["https://a.com/"]);
        let llm = FakeLlm::new(vec![Ok(summary(
            "T",
            &["lang", "lang/rust", "Lang/Rust", "food"],
            &[],
        ))]);
        run(&mut db, &llm).await.unwrap();
        let tags = db.tags().unwrap();
        let tagged: Vec<&str> = db
            .tag_links()
            .unwrap()
            .iter()
            .map(|(_, id)| tags.iter().find(|t| t.id == *id).unwrap().path.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(tagged, ["food", "lang/rust"]);
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

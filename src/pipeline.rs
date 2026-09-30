use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};

use crate::config::{Config, TagConfig};
use crate::db::{Db, EmbeddingKind, PageResult, PendingPage, page_embedding_text, tag_embedding_text};
use crate::embed::{BATCH_SIZE, Embedder, TagIndex};
use crate::extract::{extract, truncate_chars};
use crate::fetch::Fetcher;
use crate::llm::{LangMode, Llm, PageRejected, PageSummary, SummaryRequest};
use crate::tags::{CountedTag, count_tree, format_vocabulary, is_offerable, most_used, normalize_name};

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
    fn note_error(&mut self, what: &str, consequence: &str, err: &anyhow::Error) {
        if self.errors == 0 {
            println!(
                "Warning: {what} failed ({err:#}). Continuing: {consequence}. \
                 Further embedding errors are counted, not shown."
            );
        }
        self.errors += 1;
    }
}

/// Processes pending pages one at a time. Each page is saved as soon as it
/// is done, so an interrupted run continues where it stopped. Without
/// embeddings, the model is shown the most used tags instead of the most
/// relevant ones.
pub async fn process_pending<F: Fetcher, L: Llm, E: Embedder>(
    db: &mut Db,
    fetcher: &F,
    llm: &L,
    mut embeddings: Option<&mut Embeddings<'_, E>>,
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
        match process_one(db, fetcher, llm, embeddings.as_deref_mut(), config, lang, page).await {
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

async fn process_one<F: Fetcher, L: Llm, E: Embedder>(
    db: &mut Db,
    fetcher: &F,
    llm: &L,
    mut embeddings: Option<&mut Embeddings<'_, E>>,
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
    let mut shown = None;
    if let Some(e) = embeddings
        .as_deref_mut()
        .filter(|e| e.index.len() > 0 || !e.unindexed_tags.is_empty())
    {
        // Tags whose vectors failed on an earlier page go along with this
        // request, so this page can already be shown them.
        let mut texts: Vec<String> = e.unindexed_tags.iter().map(|(_, text)| text.clone()).collect();
        texts.push(format!(
            "{title}\n{}",
            truncate_chars(&extracted.text, e.page_chars)
        ));
        match e.embedder.embed(&texts).await {
            Ok(mut vectors) => {
                let vector = vectors.pop().expect("one vector per text");
                let model = e.embedder.model().to_string();
                for ((id, _), tag_vector) in std::mem::take(&mut e.unindexed_tags).into_iter().zip(vectors) {
                    db.put_embedding(EmbeddingKind::Tag, id, &model, &tag_vector)?;
                    e.index.insert(id, tag_vector);
                }
                shown = Some(relevant_tags(
                    &counted,
                    &e.index,
                    &vector,
                    config.tags.vocabulary_limit,
                ));
            }
            Err(err) => e.note_error(
                "embedding a page",
                "that page is shown the most used tags instead of the most relevant ones",
                &err,
            ),
        }
    }
    let shown = shown.unwrap_or_else(|| most_used(&counted, config.tags.vocabulary_limit));
    let vocabulary = format_vocabulary(&shown);

    let request = SummaryRequest {
        url: &page.url,
        title,
        text: &extracted.text,
        lang,
        detected_lang,
        vocabulary: &vocabulary,
        tags: &config.tags,
    };
    let summary = llm.summarize(&request).await.map_err(|err| {
        if err.downcast_ref::<PageRejected>().is_some() {
            PageError::Page(err)
        } else {
            PageError::Fatal(err.context("LLM request failed"))
        }
    })?;

    let ResolvedTags {
        page_tags: tags,
        created,
    } = resolve_tags(db, &summary, &config.tags)?;
    let title = summary.title.trim();
    let text = summary.summary.trim();
    let language = summary.language.trim().to_ascii_lowercase();
    db.save_result(
        page.id,
        &PageResult {
            title,
            summary: text,
            lang: (!language.is_empty()).then_some(language.as_str()),
            tags: &tags,
        },
    )
    .context("saving result")?;

    // New tags must be searchable for the next page; the page's own vector is
    // kept for the end-of-run tag reconciliation. The page is already saved,
    // so a failure here only costs vectors: the tags are retried with the
    // next page, and the next run fills in anything still missing.
    if let Some(e) = embeddings {
        let mut tags = std::mem::take(&mut e.unindexed_tags);
        tags.extend(created);
        let mut texts: Vec<String> = tags.iter().map(|(_, text)| text.clone()).collect();
        texts.push(page_embedding_text(title, text));
        let mut vectors = match e.embedder.embed(&texts).await {
            Ok(vectors) => vectors,
            Err(err) => {
                e.note_error(
                    "storing vectors for new tags",
                    "the tags are retried with the next page, and the next run fills in missing vectors",
                    &err,
                );
                e.unindexed_tags = tags;
                return Ok(title.to_string());
            }
        };
        let model = e.embedder.model().to_string();
        db.put_embedding(
            EmbeddingKind::Page,
            page.id,
            &model,
            &vectors.pop().expect("one vector per text"),
        )?;
        for ((id, _), vector) in tags.iter().zip(vectors) {
            db.put_embedding(EmbeddingKind::Tag, *id, &model, &vector)?;
            e.index.insert(*id, vector);
        }
    }
    Ok(title.to_string())
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

    use super::*;
    use crate::fetch::FetchedPage;
    use crate::llm::NewTag;

    struct FakeFetcher;

    impl Fetcher for FakeFetcher {
        async fn fetch(&self, url: &str) -> Result<FetchedPage> {
            if url.contains("broken") {
                bail!("HTTP 404 Not Found");
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
                Err("reject") => Err(PageRejected("too long".into()).into()),
                Err(other) => bail!("{other}"),
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
            &Config::default(),
            &LangMode::English,
            None,
        )
        .await
    }

    async fn run_with_embeddings(db: &mut Db, llm: &FakeLlm, config: &Config) -> Result<Stats> {
        let mut embeddings = Embeddings::prepare(db, &FakeEmbedder, 2000).await.unwrap();
        process_pending(
            db,
            &FakeFetcher,
            llm,
            Some(&mut embeddings),
            config,
            &LangMode::English,
            None,
        )
        .await
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
        assert_eq!(stats, Stats { done: 2, failed: 1 });

        // The second page was shown the tags the first page created.
        let seen = llm.seen_vocab.borrow();
        assert_eq!(seen[0], "");
        assert_eq!(
            seen[1],
            "memory-safety (1) - first used for: Rust intro\nrust-programming (1) - The Rust language"
        );

        // The reserved status/ tag from the model was dropped; paths became flat names.
        assert_eq!(
            page_tag_names(&db),
            ["memory-safety", "rust-programming", "rust-programming"]
        );
        assert_eq!(db.done_pages().unwrap()[0].lang.as_deref(), Some("en"));
        assert_eq!(db.status_counts().unwrap().failed, 1);
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
            &Config::default(),
            &LangMode::English,
            None,
        )
        .await
        .unwrap();
        assert_eq!(stats, Stats { done: 2, failed: 0 });
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
            &Config::default(),
            &LangMode::English,
            None,
        )
        .await
        .unwrap();
        assert_eq!(stats, Stats { done: 2, failed: 0 });
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
            &config,
            &LangMode::English,
            None,
        )
        .await
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

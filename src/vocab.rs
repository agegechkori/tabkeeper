//! Building the tag vocabulary from the whole archive, then tagging every page
//! from it. Pages are clustered by what they're about, the model names each
//! cluster, topics already used on several pages join the vocabulary, and each
//! page then gets its tags from the vocabulary tags nearest to it.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::Result;
use serde::Deserialize;
use serde_json::json;

use crate::db::{AnalyzedPage, Db};
use crate::embed::{BATCH_SIZE, Embedder, similarity};
use crate::extract::truncate_chars;
use crate::llm::Reviewer;
use crate::meanings;
use crate::tags::normalize_name;

/// Member pages shown to the model when it names a cluster.
const CLUSTER_SAMPLE: usize = 10;
/// Summary characters shown for a page when it is tagged.
const SUMMARY_CHARS: usize = 500;

#[derive(Debug, Clone)]
pub struct Settings {
    /// Pages at least this alike (average cosine similarity) form a cluster.
    pub cluster_similarity: f32,
    /// A tag needs at least this many pages to stay in the vocabulary.
    pub min_support: usize,
    /// Vocabulary tags offered to the model for each page.
    pub candidates: usize,
    /// Most tags a page gets.
    pub max_tags: usize,
}

/// Where a vocabulary tag came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// A topic pages were tagged with that several pages share.
    Topic,
    /// The model's name for a cluster of pages.
    Cluster,
}

#[derive(Debug, Clone)]
pub struct VocabTag {
    pub name: String,
    pub description: String,
    pub source: Source,
    /// Pages tagged with it in the end.
    pub pages: usize,
}

/// What retagging found and decided.
#[derive(Debug, Default)]
pub struct Retag {
    pub vocabulary: Vec<VocabTag>,
    /// Tags offered or chosen but used on too few pages in the end.
    pub dropped: Vec<String>,
    /// Each done page's topic tags now, without locked ones.
    pub before: HashMap<i64, Vec<String>>,
    /// (page, its new tags as (name, description)), for the pages the model
    /// tagged. Pages whose request failed keep their tags. Locked tags stay
    /// on their pages and aren't listed.
    pub assignments: Vec<(i64, Vec<(String, String)>)>,
    pub untagged: usize,
    pub failed_requests: Vec<String>,
    /// Words asked whether they have several meanings.
    pub meaning_checks: usize,
    /// Each word checked for several meanings, with the titles of the first
    /// page of each group it was shown.
    pub checked: Vec<(String, Vec<String>)>,
    /// One-word topics found to have several meanings, with the tags they
    /// became: the word itself for the meaning with the most pages, then a
    /// qualified tag for each other meaning.
    pub ambiguous: Vec<(String, Vec<String>)>,
}

const CLUSTER_SYSTEM: &str = "You name a group of pages from a personal archive. The numbered pages were grouped because they are about similar things. Give the one tag for the topic most of them share: the broad topic someone would file them under, like bread, investing, japan, astronomy or machine-learning.

- The tag names a topic, not an aspect of one: astronomy, not astronomy-history; birds, not bird-species.
- Use lowercase kebab-case English. If the word has several common meanings, the most common one keeps the plain word (python, rust, java) and only the others are qualified: python-snake, rust-corrosion, java-island.
- If one of the existing tags names that topic, use its exact name.
- fits lists the numbers of the pages the tag describes.
- If the pages have nothing useful in common, give an empty tag.

Reply with a JSON object: {\"tag\": \"...\", \"description\": \"one line on what the tag covers\", \"fits\": [1, 2, 3]}.";

const CLASSIFY_SYSTEM: &str = "You tag a page of a personal archive with tags from a list. Pick the tags that describe what the page is about.

- Pick 1 to {max} tags, the broadest first. Pick none only if no tag fits at all.
- A broad tag fits a page about one of its kinds: vehicles fits a page about trucks; sport fits a page about tennis.
- A tag fits only in its own meaning, as its description says: a tag for the programming language never fits a page about the snake.
- Never pick two tags that mean the same thing.
- Use each name exactly as listed, without its description.

Reply with a JSON object: {\"tags\": [\"name\", ...]}.";

#[derive(Deserialize)]
struct ClusterReply {
    #[serde(default)]
    tag: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    fits: Vec<usize>,
}

/// Groups of pages shown when asking whether a word has several meanings.
const MAX_MEANING_GROUPS: usize = 4;

#[derive(Deserialize)]
struct ClassifyReply {
    #[serde(default)]
    tags: Vec<String>,
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
    v
}

async fn embed_all<E: Embedder>(embedder: &E, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(BATCH_SIZE) {
        out.extend(embedder.embed(chunk).await?.into_iter().map(unit));
    }
    Ok(out)
}

/// Average-linkage agglomerative clustering: the two most alike clusters are
/// joined until no two are at least `threshold` alike. Pages like no other
/// stay on their own. Each cluster remembers its most alike other cluster, so
/// a join only rescans the clusters that pointed at the joined two: about
/// O(n²) time, and O(n²) memory for the similarities.
pub fn agglomerate(vectors: &[Vec<f32>], threshold: f32) -> Vec<Vec<usize>> {
    let n = vectors.len();
    let mut members: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    let mut sim = vec![vec![0f32; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let s = similarity(&vectors[i], &vectors[j]);
            sim[i][j] = s;
            sim[j][i] = s;
        }
    }
    // The most alike other live cluster of each live cluster.
    let nearest = |a: usize, sim: &[Vec<f32>], members: &[Option<Vec<usize>>]| {
        (0..n)
            .filter(|&b| b != a && members[b].is_some())
            .map(|b| (b, sim[a][b]))
            .max_by(|x, y| x.1.total_cmp(&y.1).then(y.0.cmp(&x.0)))
    };
    let mut best: Vec<Option<(usize, f32)>> = (0..n).map(|a| nearest(a, &sim, &members)).collect();
    while let Some((a, b)) = (0..n)
        .filter_map(|a| best[a].map(|(b, s)| (a, b, s)))
        .max_by(|x, y| x.2.total_cmp(&y.2).then((y.0, y.1).cmp(&(x.0, x.1))))
        .filter(|(_, _, s)| *s >= threshold)
        .map(|(a, b, _)| (a.min(b), a.max(b)))
    {
        let (na, nb) = (
            members[a].as_ref().map_or(0, Vec::len) as f32,
            members[b].as_ref().map_or(0, Vec::len) as f32,
        );
        for k in (0..n).filter(|&k| k != a && k != b && members[k].is_some()) {
            let merged = (na * sim[a][k] + nb * sim[b][k]) / (na + nb);
            sim[a][k] = merged;
            sim[k][a] = merged;
        }
        let moved = members[b].take().unwrap_or_default();
        members[a].get_or_insert_with(Vec::new).extend(moved);
        best[b] = None;
        best[a] = nearest(a, &sim, &members);
        for k in (0..n).filter(|&k| k != a && members[k].is_some()) {
            best[k] = match best[k] {
                Some((p, _)) if p == a || p == b => nearest(k, &sim, &members),
                Some((p, s)) if sim[k][a] > s || (sim[k][a] == s && a < p) => Some((a, sim[k][a])),
                other => other,
            };
        }
    }
    let mut clusters: Vec<Vec<usize>> = members.into_iter().flatten().collect();
    clusters.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
    clusters
}

/// The tag for a meaning of `word` other than its most common one: always
/// `word-…`. A vocabulary tag of that form already made for these pages (a
/// cluster's name) is reused, so one meaning doesn't get two names. Otherwise
/// the model's name is used, prefixed with the word if it lacks it; if the
/// model gave the plain word, the qualifier comes from the description.
fn qualified_name(
    word: &str,
    name: &str,
    description: &str,
    pages: &[usize],
    members: &BTreeMap<String, Vec<usize>>,
) -> Option<String> {
    let prefix = format!("{word}-");
    let existing = members
        .iter()
        .filter(|(tag, _)| tag.starts_with(&prefix))
        .map(|(tag, list)| (pages.iter().filter(|p| list.contains(p)).count(), tag))
        .filter(|(shared, _)| *shared > 0)
        .max_by_key(|(shared, _)| *shared);
    if let Some((_, tag)) = existing {
        return Some(tag.clone());
    }
    if name.starts_with(&prefix) {
        return Some(name.to_string());
    }
    if !name.is_empty() && name != word {
        return normalize_name(&format!("{prefix}{name}"));
    }
    const SKIP: [&str; 9] = ["the", "a", "an", "of", "and", "in", "for", "its", "to"];
    let qualifier: Vec<String> = description
        .split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|w| !w.is_empty() && !SKIP.contains(&w.as_str()) && !w.starts_with(word))
        .take(2)
        .collect();
    if qualifier.is_empty() {
        return None;
    }
    normalize_name(&format!("{prefix}{}", qualifier.join("-")))
}

fn page_line(page: &AnalyzedPage) -> String {
    if page.tags.is_empty() {
        page.title.clone()
    } else {
        format!("{} (topics: {})", page.title, page.tags.join(", "))
    }
}

/// The vocabulary as the classifier sees it: each tag's name, description
/// and vector.
struct Offer<'a> {
    names: &'a [String],
    descriptions: &'a [String],
    vectors: &'a [Vec<f32>],
    system: &'a str,
    schema: &'a serde_json::Value,
}

/// The page's tags from the vocabulary tags nearest to it, except `exclude`,
/// as the model picks them. `Err` says why the request failed.
async fn classify<R: Reviewer>(
    reviewer: &R,
    offer: &Offer<'_>,
    page: &AnalyzedPage,
    vector: &[f32],
    settings: &Settings,
    exclude: &[String],
) -> Result<Vec<String>, String> {
    let mut ranked: Vec<(f32, usize)> = offer
        .vectors
        .iter()
        .enumerate()
        .filter(|(i, _)| !exclude.contains(&offer.names[*i]))
        .map(|(i, t)| (similarity(vector, t), i))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let offered: Vec<usize> = ranked.iter().take(settings.candidates).map(|(_, i)| *i).collect();
    let mut user = format!(
        "Page: {}\n{}\n\nTags to choose from:\n",
        page.title,
        truncate_chars(&page.summary, SUMMARY_CHARS)
    );
    for &i in &offered {
        user.push_str(&format!("- {}: {}\n", offer.names[i], offer.descriptions[i]));
    }
    let reply = reviewer
        .ask_json(offer.system, &user, "page_tags", offer.schema.clone())
        .await
        .map_err(|err| format!("{err:#}"))?;
    let picked = serde_json::from_value::<ClassifyReply>(reply)
        .map_err(|_| "a page's tags were unusable".to_string())?
        .tags;
    let mut tags: Vec<String> = Vec::new();
    for raw in picked {
        // A small model sometimes copies the whole line, "name: description".
        let raw = raw.split(':').next().unwrap_or_default();
        if let Some(name) = normalize_name(raw)
            && offered.iter().any(|&i| offer.names[i] == name)
            && !tags.contains(&name)
            && tags.len() < settings.max_tags
        {
            tags.push(name);
        }
    }
    Ok(tags)
}

/// What the model said about a word with several meanings.
struct Split {
    /// What the word names in its most common meaning, the one keeping it.
    main_description: String,
    /// Each other meaning: its qualified tag, description and pages.
    others: Vec<(String, String, Vec<usize>)>,
}

/// Asks whether `word`, on the pages `list` (indexes into `pages`), names
/// unrelated things. `None` when it doesn't: one meaning (remembered, and
/// asked again only once the word has twice the pages), all pages in one
/// cluster, no other meaning that could be named, or a failed request
/// (recorded in `out`). A one-meaning verdict is remembered only if
/// `remember`.
#[allow(clippy::too_many_arguments)]
async fn check_word<R: Reviewer>(
    db: &Db,
    reviewer: &R,
    word: &str,
    list: &[usize],
    cluster_of: &[usize],
    pages: &[AnalyzedPage],
    members: &BTreeMap<String, Vec<usize>>,
    remember: bool,
    out: &mut Retag,
) -> Result<Option<Split>> {
    let key = format!("meanings:{word}");
    if let Some(n) = db
        .tag_decision(&key)?
        .and_then(|d| d.strip_prefix("same:").and_then(|n| n.parse::<usize>().ok()))
        && list.len() < n * 2
    {
        return Ok(None);
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &i in list {
        groups.entry(cluster_of[i]).or_default().push(i);
    }
    if groups.len() < 2 {
        return Ok(None);
    }
    let mut groups: Vec<Vec<usize>> = groups.into_values().collect();
    groups.sort_by_key(|g| std::cmp::Reverse(g.len()));
    // Pages in clusters beyond the first few go with the largest group.
    while groups.len() > MAX_MEANING_GROUPS {
        let extra = groups.pop().unwrap_or_default();
        groups[0].extend(extra);
    }
    let shown: Vec<Vec<&AnalyzedPage>> = groups
        .iter()
        .map(|g| {
            g.iter()
                .take(meanings::PAGES_PER_GROUP)
                .map(|&i| &pages[i])
                .collect()
        })
        .collect();
    out.meaning_checks += 1;
    out.checked.push((
        word.to_string(),
        shown
            .iter()
            .filter_map(|g| g.first().map(|p| p.title.clone()))
            .collect(),
    ));
    let things = match meanings::check(reviewer, word, &shown).await {
        Ok(meanings::Verdict::Same) => {
            if remember {
                db.set_tag_decision(&key, &format!("same:{}", list.len()), "llm")?;
            }
            return Ok(None);
        }
        Ok(meanings::Verdict::Several(things)) => things,
        Err(err) => {
            out.failed_requests.push(err);
            return Ok(None);
        }
    };
    // The meaning with the most pages keeps the plain word, so a false
    // split only takes the word off the other groups' pages; each other
    // meaning gets its qualified tag.
    let pages_of =
        |thing: &(String, String, Vec<usize>)| thing.2.iter().map(|&g| groups[g].len()).sum::<usize>();
    let main = (0..things.len())
        .max_by_key(|&t| pages_of(&things[t]))
        .unwrap_or(0);
    let mut others: Vec<(String, String, Vec<usize>)> = Vec::new();
    for (t, (name, description, member_groups)) in things.iter().enumerate() {
        if t == main {
            continue;
        }
        let thing_pages: Vec<usize> = member_groups
            .iter()
            .flat_map(|&g| groups[g].iter().copied())
            .collect();
        let Some(name) = qualified_name(word, name, description, &thing_pages, members) else {
            continue;
        };
        match others.iter_mut().find(|(n, _, _)| *n == name) {
            Some((_, _, list)) => list.extend(thing_pages),
            None => others.push((name, description.clone(), thing_pages)),
        }
    }
    if others.is_empty() {
        // No other meaning could be named: the word stays as it was.
        return Ok(None);
    }
    Ok(Some(Split {
        main_description: things[main].1.clone(),
        others,
    }))
}

/// Builds the vocabulary from the whole archive and tags every done page
/// from it. Only the model's "one meaning" verdicts are saved, so they aren't
/// asked again; `Retag::assignments` says which tags to save.
pub async fn retag<R: Reviewer, E: Embedder>(
    db: &Db,
    reviewer: &R,
    embedder: &E,
    settings: &Settings,
    note: impl Fn(&str),
) -> Result<Retag> {
    let mut out = Retag::default();
    let pages = db.analyzed_pages()?;
    if pages.is_empty() {
        return Ok(out);
    }
    let texts: Vec<String> = pages
        .iter()
        .map(|p| {
            format!(
                "Title: {}\nSummary: {}\nTopics: {}",
                p.title,
                p.summary,
                p.tags.join("; ")
            )
        })
        .collect();
    let page_vectors = embed_all(embedder, &texts).await?;

    // Topics several pages already share.
    let topics = db.topic_tags()?;
    let descriptions: HashMap<String, String> = topics
        .iter()
        .map(|t| (t.name.clone(), t.description.clone().unwrap_or_default()))
        .collect();
    let locked: HashSet<String> = topics
        .iter()
        .filter(|t| t.locked)
        .map(|t| t.name.clone())
        .collect();
    out.before = pages
        .iter()
        .map(|p| {
            let tags = p.tags.iter().filter(|t| !locked.contains(*t)).cloned().collect();
            (p.id, tags)
        })
        .collect();
    let mut topic_counts: BTreeMap<&str, usize> = BTreeMap::new();
    // Tags some page has as its broad topic, its first tag. A tag only ever
    // given after it names an aspect of other topics (history, taxonomy) and
    // doesn't become a vocabulary tag of its own.
    let mut first_tags: HashSet<&str> = HashSet::new();
    for page in &pages {
        for tag in &page.tags {
            *topic_counts.entry(tag.as_str()).or_default() += 1;
        }
        if let Some(first) = page.tags.first() {
            first_tags.insert(first.as_str());
        }
    }
    let mut vocabulary: BTreeMap<String, VocabTag> = BTreeMap::new();
    // Pages each vocabulary tag was made for, by index.
    let mut members: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, page) in pages.iter().enumerate() {
        for tag in &page.tags {
            if first_tags.contains(tag.as_str())
                && topic_counts
                    .get(tag.as_str())
                    .is_some_and(|n| *n >= settings.min_support)
            {
                members.entry(tag.clone()).or_default().push(i);
            }
        }
    }
    for name in members.keys() {
        vocabulary.insert(
            name.to_string(),
            VocabTag {
                name: name.to_string(),
                description: descriptions.get(name).cloned().unwrap_or_default(),
                source: Source::Topic,
                pages: 0,
            },
        );
    }

    // A name for each cluster big enough to deserve a tag.
    let all_clusters = agglomerate(&page_vectors, settings.cluster_similarity);
    let mut cluster_of = vec![0; pages.len()];
    for (c, list) in all_clusters.iter().enumerate() {
        for &i in list {
            cluster_of[i] = c;
        }
    }
    let clusters: Vec<Vec<usize>> = all_clusters
        .into_iter()
        .filter(|c| c.len() >= settings.min_support)
        .collect();
    note(&format!(
        "Building the vocabulary: {} clusters of pages, {} shared topics.",
        clusters.len(),
        vocabulary.len()
    ));
    let cluster_schema = json!({
        "type": "object", "additionalProperties": false, "required": ["tag", "description", "fits"],
        "properties": {
            "tag": {"type": "string"}, "description": {"type": "string"},
            "fits": {"type": "array", "items": {"type": "integer"}}
        }
    });
    for cluster in &clusters {
        let sample: Vec<&AnalyzedPage> = cluster.iter().take(CLUSTER_SAMPLE).map(|&i| &pages[i]).collect();
        let existing: Vec<&str> = vocabulary.keys().map(String::as_str).collect();
        let mut user = format!(
            "Existing tags: {}\n\nPages:\n",
            if existing.is_empty() {
                "(none yet)".to_string()
            } else {
                existing.join(", ")
            }
        );
        for (i, page) in sample.iter().enumerate() {
            user.push_str(&format!("{}. {}\n", i + 1, page_line(page)));
        }
        let reply = match reviewer
            .ask_json(CLUSTER_SYSTEM, &user, "cluster_tag", cluster_schema.clone())
            .await
        {
            Ok(reply) => reply,
            Err(err) => {
                out.failed_requests.push(format!("{err:#}"));
                continue;
            }
        };
        let Ok(reply) = serde_json::from_value::<ClusterReply>(reply) else {
            out.failed_requests.push("a cluster name was unusable".into());
            continue;
        };
        let Some(name) = normalize_name(&reply.tag) else {
            continue;
        };
        let fitting: Vec<usize> = reply
            .fits
            .iter()
            .filter(|&&i| i >= 1 && i <= sample.len())
            .map(|&i| cluster[i - 1])
            .collect::<BTreeSet<usize>>()
            .into_iter()
            .collect();
        if fitting.len() < settings.min_support.min(sample.len()) {
            continue;
        }
        let list = members.entry(name.clone()).or_default();
        for i in fitting {
            if !list.contains(&i) {
                list.push(i);
            }
        }
        // A cluster's description was written for its pages, so it replaces
        // the one a topic got from the first page it was used on.
        let description = reply.description.trim().to_string();
        match vocabulary.get_mut(&name) {
            Some(tag) if !description.is_empty() => tag.description = description,
            Some(_) => {}
            None => {
                vocabulary.insert(
                    name.clone(),
                    VocabTag {
                        name,
                        description,
                        source: Source::Cluster,
                        pages: 0,
                    },
                );
            }
        }
    }
    if vocabulary.is_empty() {
        return Ok(out);
    }

    // A one-word topic whose pages fall into different clusters may be one
    // word for unrelated things (python the language and the snake): the
    // model says whether it is, and each meaning gets a qualified tag.
    let words: Vec<String> = vocabulary
        .values()
        .filter(|t| t.source == Source::Topic && !t.name.contains('-'))
        .map(|t| t.name.clone())
        .collect();
    for word in words {
        let Some(list) = members.get(&word).cloned() else {
            continue;
        };
        let Some(split) = check_word(
            db,
            reviewer,
            &word,
            &list,
            &cluster_of,
            &pages,
            &members,
            true,
            &mut out,
        )
        .await?
        else {
            continue;
        };
        let mut names = vec![word.clone()];
        for (name, description, thing_pages) in split.others {
            vocabulary.entry(name.clone()).or_insert(VocabTag {
                name: name.clone(),
                description,
                source: Source::Topic,
                pages: 0,
            });
            // So a later check of the word reuses this name for these pages.
            members.entry(name.clone()).or_default().extend(thing_pages);
            names.push(name);
        }
        // The word now names that meaning: its description says so, so a page
        // about another meaning isn't given it.
        if let Some(tag) = vocabulary.get_mut(&word)
            && !split.main_description.is_empty()
        {
            tag.description = format!("{word}: {}", split.main_description);
        }
        out.ambiguous.push((word, names));
    }

    // Each page's tags, from the vocabulary tags nearest to it.
    let names: Vec<String> = vocabulary.keys().cloned().collect();
    let descriptions: Vec<String> = vocabulary.values().map(|t| t.description.clone()).collect();
    let tag_texts: Vec<String> = vocabulary
        .values()
        .map(|t| format!("{}: {}", t.name.replace('-', " "), t.description))
        .collect();
    let tag_vectors = embed_all(embedder, &tag_texts).await?;
    note(&format!(
        "Tagging {} pages from a vocabulary of {} tags.",
        pages.len(),
        names.len()
    ));
    let system = CLASSIFY_SYSTEM.replace("{max}", &settings.max_tags.to_string());
    let classify_schema = json!({
        "type": "object", "additionalProperties": false, "required": ["tags"],
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}}
    });
    let offer = Offer {
        names: &names,
        descriptions: &descriptions,
        vectors: &tag_vectors,
        system: &system,
        schema: &classify_schema,
    };
    // None for a page whose request failed: it keeps its tags.
    let mut chosen: Vec<Option<Vec<String>>> = Vec::with_capacity(pages.len());
    for (page, vector) in pages.iter().zip(&page_vectors) {
        match classify(reviewer, &offer, page, vector, settings, &[]).await {
            Ok(tags) => chosen.push(Some(tags)),
            Err(err) => {
                out.failed_requests.push(err);
                chosen.push(None);
            }
        }
    }

    // The classifier can give a word's tag to a page about another of its
    // meanings (rust the language to a page on corrosion), also when the
    // pages' own tags never showed the word in two meanings. A one-word tag
    // given to pages in different clusters is checked again with those pages.
    // A page of another meaning gets its qualified tag instead, if the
    // vocabulary has it; otherwise it is tagged again without the word (a
    // tag made for one stray page would be dropped and leave it untagged).
    // Each page tagged again: its tags before, and the words taken off it.
    let mut retry: BTreeMap<usize, (Vec<String>, Vec<String>)> = BTreeMap::new();
    let words: Vec<String> = vocabulary
        .keys()
        .filter(|name| !name.contains('-') && !locked.contains(*name))
        .cloned()
        .collect();
    for word in words {
        let list: Vec<usize> = chosen
            .iter()
            .enumerate()
            .filter(|(_, tags)| tags.as_ref().is_some_and(|t| t.contains(&word)))
            .map(|(i, _)| i)
            .collect();
        // A word just found to have several meanings may show only one of
        // them here; that isn't remembered as the word having one meaning.
        let split_before = out.ambiguous.iter().any(|(w, _)| *w == word);
        let Some(split) = check_word(
            db,
            reviewer,
            &word,
            &list,
            &cluster_of,
            &pages,
            &members,
            !split_before,
            &mut out,
        )
        .await?
        else {
            continue;
        };
        let mut names = vec![word.clone()];
        for (name, _, thing_pages) in split.others {
            let known = vocabulary.contains_key(&name);
            for i in thing_pages {
                let Some(Some(tags)) = chosen.get_mut(i) else {
                    continue;
                };
                let Some(at) = tags.iter().position(|t| *t == word) else {
                    continue;
                };
                if known && !tags.contains(&name) {
                    tags[at] = name.clone();
                } else {
                    let before = tags.clone();
                    tags.remove(at);
                    if !known {
                        retry
                            .entry(i)
                            .or_insert((before, Vec::new()))
                            .1
                            .push(word.clone());
                    }
                }
            }
            names.push(name);
        }
        match out.ambiguous.iter_mut().find(|(w, _)| *w == word) {
            Some((_, known)) => {
                for name in names {
                    if !known.contains(&name) {
                        known.push(name);
                    }
                }
            }
            None => out.ambiguous.push((word, names)),
        }
    }
    // A page loses the word only for another tag that fits: a broad word
    // wrongly found to have several meanings (running on a page about
    // sneakers) stays on a page nothing else fits.
    for (i, (before, exclude)) in retry {
        match classify(reviewer, &offer, &pages[i], &page_vectors[i], settings, &exclude).await {
            Ok(tags) if !tags.is_empty() => chosen[i] = Some(tags),
            Ok(_) => chosen[i] = Some(before),
            // The page keeps its other tags.
            Err(err) => out.failed_requests.push(err),
        }
    }

    // Tags too few pages chose are dropped.
    let mut counts: HashMap<String, usize> = HashMap::new();
    for tags in chosen.iter().flatten() {
        for t in tags {
            *counts.entry(t.clone()).or_default() += 1;
        }
    }
    let kept: HashSet<String> = counts
        .iter()
        .filter(|(_, n)| **n >= settings.min_support)
        .map(|(t, _)| t.to_string())
        .collect();
    out.dropped = names.iter().filter(|n| !kept.contains(*n)).cloned().collect();
    for (page, tags) in pages.iter().zip(chosen) {
        let Some(tags) = tags else { continue };
        let tags: Vec<(String, String)> = tags
            .into_iter()
            .filter(|t| kept.contains(t) && !locked.contains(t))
            .map(|t| {
                let description = vocabulary[&t].description.clone();
                (t, description)
            })
            .collect();
        // A page with a locked tag keeps it.
        if tags.is_empty() && !page.tags.iter().any(|t| locked.contains(t)) {
            out.untagged += 1;
        }
        out.assignments.push((page.id, tags));
    }
    for tag in vocabulary.values_mut() {
        tag.pages = counts.get(&tag.name).copied().unwrap_or(0);
    }
    out.vocabulary = vocabulary
        .into_values()
        .filter(|t| kept.contains(&t.name))
        .collect();
    out.vocabulary
        .sort_by(|a, b| b.pages.cmp(&a.pages).then(a.name.cmp(&b.name)));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn builds_the_vocabulary_and_qualifies_a_word_with_two_meanings() {
        use crate::reconcile::tests::{FakeReviewer, archive, no_names};
        let db = archive(&[
            (&["coffee"], [1.0, 0.0, 0.0]),
            (&["coffee"], [1.0, 0.0, 0.0]),
            (&["coffee"], [1.0, 0.0, 0.0]),
            (&["python"], [0.0, 1.0, 0.0]),
            (&["python"], [0.0, 1.0, 0.0]),
            (&["python"], [0.0, 0.0, 1.0]),
        ]);
        let reviewer = FakeReviewer::new(vec![
            // The test embedder makes every page unlike the others: no
            // clusters, so the topics on 2+ pages are the vocabulary, and
            // each page is its own group when checking meanings.
            json!({"groups": [], "same_meaning": true}),
            json!({"groups": [], "same_meaning": false}),
            json!({"referent": "the programming language"}),
            json!({"referent": "the programming language"}),
            json!({"referent": "the snake"}),
            json!({"groups": [
                {"group": 1, "tag": "python-programming", "description": "Python, the language."},
                {"group": 2, "tag": "python-programming", "description": "Python, the language."},
                {"group": 3, "tag": "python-snake", "description": "Pythons, the snakes."}]}),
            json!({"tags": ["coffee"]}),
            json!({"tags": ["coffee"]}),
            // The whole line copied, and a name not offered.
            json!({"tags": ["coffee: the drink", "nowhere"]}),
            json!({"tags": ["python"]}),
            json!({"tags": ["python"]}),
            json!({"tags": ["python-snake"]}),
            // After classification python is checked again, with the two
            // language pages it was given to.
            json!({"groups": [], "same_meaning": true}),
        ]);
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 2,
            candidates: 8,
            max_tags: 3,
        };
        let out = retag(&db, &reviewer, &no_names(), &settings, |_| {})
            .await
            .unwrap();
        assert_eq!(out.meaning_checks, 3);
        // The meaning with the most pages keeps the plain word.
        assert_eq!(
            out.ambiguous,
            [(
                "python".to_string(),
                vec!["python".to_string(), "python-snake".to_string()]
            )]
        );
        let tags: Vec<Vec<&str>> = out
            .assignments
            .iter()
            .map(|(_, t)| t.iter().map(|(n, _)| n.as_str()).collect())
            .collect();
        // python-snake is on one page, too few to keep.
        assert_eq!(
            tags,
            [
                vec!["coffee"],
                vec!["coffee"],
                vec!["coffee"],
                vec!["python"],
                vec!["python"],
                vec![]
            ]
        );
        assert_eq!(out.untagged, 1);
        assert_eq!(out.dropped, ["python-snake"]);
        // The one-meaning verdict is remembered, but not python's second
        // check, which only saw the language.
        assert_eq!(
            db.tag_decision("meanings:coffee").unwrap().as_deref(),
            Some("same:3")
        );
        assert_eq!(db.tag_decision("meanings:python").unwrap(), None);
    }

    #[tokio::test]
    async fn a_tag_never_given_first_stays_out_of_the_vocabulary() {
        use crate::reconcile::tests::{FakeReviewer, archive, no_names};
        // history is on every page, but never as the page's broad topic.
        let db = archive(&[
            (&["coffee", "history"], [1.0, 0.0, 0.0]),
            (&["coffee", "history"], [1.0, 0.0, 0.0]),
            (&["bread", "history"], [0.0, 1.0, 0.0]),
            (&["bread", "history"], [0.0, 1.0, 0.0]),
        ]);
        let reviewer = FakeReviewer::new(vec![
            json!({"groups": [], "same_meaning": true}),
            json!({"groups": [], "same_meaning": true}),
            json!({"tags": ["coffee"]}),
            json!({"tags": ["coffee"]}),
            json!({"tags": ["bread"]}),
            json!({"tags": ["bread"]}),
        ]);
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 2,
            candidates: 8,
            max_tags: 3,
        };
        let out = retag(&db, &reviewer, &no_names(), &settings, |_| {})
            .await
            .unwrap();
        let names: Vec<&str> = out.vocabulary.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["bread", "coffee"]);
    }

    #[tokio::test]
    async fn a_page_whose_request_fails_keeps_its_tags() {
        use crate::reconcile::tests::{FakeReviewer, archive, no_names};
        let db = archive(&[
            (&["coffee"], [1.0, 0.0, 0.0]),
            (&["coffee"], [1.0, 0.0, 0.0]),
            (&["coffee"], [1.0, 0.0, 0.0]),
        ]);
        let reviewer = FakeReviewer::new(vec![
            json!({"groups": [], "same_meaning": true}),
            json!({"tags": ["coffee"]}),
            json!({"tags": ["coffee"]}),
            json!({"tags": 5}),
        ]);
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 2,
            candidates: 8,
            max_tags: 3,
        };
        let out = retag(&db, &reviewer, &no_names(), &settings, |_| {})
            .await
            .unwrap();
        assert_eq!(out.failed_requests.len(), 1);
        assert_eq!(out.assignments.len(), 2, "the third page isn't retagged");
        assert_eq!(out.untagged, 0);
    }

    /// Joining the most alike pair by scanning every pair, as agglomerate did
    /// before it remembered each cluster's nearest one.
    fn agglomerate_by_scanning(vectors: &[Vec<f32>], threshold: f32) -> Vec<Vec<usize>> {
        let n = vectors.len();
        let mut members: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
        let mut sim: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..n).map(|j| similarity(&vectors[i], &vectors[j])).collect())
            .collect();
        loop {
            let mut best: Option<(usize, usize, f32)> = None;
            for a in (0..n).filter(|&a| members[a].is_some()) {
                for b in (a + 1..n).filter(|&b| members[b].is_some()) {
                    if best.is_none_or(|(_, _, s)| sim[a][b] > s) {
                        best = Some((a, b, sim[a][b]));
                    }
                }
            }
            let Some((a, b, _)) = best.filter(|(_, _, s)| *s >= threshold) else {
                break;
            };
            let na = members[a].as_ref().map_or(0, Vec::len) as f32;
            let nb = members[b].as_ref().map_or(0, Vec::len) as f32;
            for k in (0..n).filter(|&k| k != a && k != b && members[k].is_some()) {
                let merged = (na * sim[a][k] + nb * sim[b][k]) / (na + nb);
                sim[a][k] = merged;
                sim[k][a] = merged;
            }
            let moved = members[b].take().unwrap_or_default();
            members[a].get_or_insert_with(Vec::new).extend(moved);
        }
        let mut clusters: Vec<Vec<usize>> = members.into_iter().flatten().collect();
        clusters.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
        clusters
    }

    #[test]
    fn clusters_match_scanning_every_pair() {
        let mut seed: u64 = 7;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        // Points near a few centres, so pages form clusters of several sizes.
        let centres: Vec<Vec<f32>> = (0..6).map(|_| (0..8).map(|_| next()).collect()).collect();
        let vectors: Vec<Vec<f32>> = (0..200)
            .map(|i| unit(centres[i % 6].iter().map(|c| c + 0.4 * next()).collect()))
            .collect();
        for threshold in [0.5, 0.7, 0.9] {
            assert_eq!(
                agglomerate(&vectors, threshold),
                agglomerate_by_scanning(&vectors, threshold),
                "threshold {threshold}"
            );
        }
    }

    #[test]
    fn other_meanings_get_the_word_as_prefix() {
        let members: BTreeMap<String, Vec<usize>> = [
            ("java-indonesia".to_string(), vec![7, 8]),
            ("java".to_string(), vec![1, 7]),
        ]
        .into();
        let name = |word, name, description, pages: &[usize]| {
            qualified_name(word, name, description, pages, &members)
        };
        // A cluster's name for these pages is reused.
        assert_eq!(
            name("java", "java-island", "the island", &[7]).as_deref(),
            Some("java-indonesia")
        );
        assert_eq!(
            name("python", "python-snake", "the snakes", &[3]).as_deref(),
            Some("python-snake")
        );
        assert_eq!(
            name("python", "snake", "the snakes", &[3]).as_deref(),
            Some("python-snake")
        );
        // The plain word: the qualifier comes from the description.
        assert_eq!(
            name("rust", "rust", "the iron oxide corrosion process", &[3]).as_deref(),
            Some("rust-iron-oxide")
        );
        assert_eq!(name("rust", "", "the rust", &[3]), None);
    }

    #[test]
    fn clusters_join_alike_pages_and_leave_outliers() {
        let v = |x: f32, y: f32, z: f32| unit(vec![x, y, z]);
        let vectors = [
            v(1.0, 0.0, 0.0),
            v(1.0, 0.1, 0.0),
            v(1.0, 0.0, 0.1),
            v(0.0, 1.0, 0.0),
            v(0.1, 1.0, 0.0),
            v(0.0, 0.0, 1.0),
        ];
        let clusters = agglomerate(&vectors, 0.9);
        assert_eq!(clusters, [vec![0, 1, 2], vec![3, 4], vec![5]]);
    }
}

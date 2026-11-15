//! Building the tag vocabulary from the whole archive, then tagging every page
//! from it. Pages are clustered by what they're about, the model names each
//! cluster, topics already used on several pages join the vocabulary, and each
//! page then gets its tags from the vocabulary tags nearest to it.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::Result;
use futures_util::stream::{self, StreamExt};
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
/// The classifier's choice for a page about none of the offered tags.
const NONE: &str = "none";
/// Summary characters shown for a page when it is tagged.
const SUMMARY_CHARS: usize = 500;
/// How much more alike pages must be to form a cluster in the first,
/// tighter pass of cluster naming than `Settings::cluster_similarity`.
const TIGHT_STEP: f32 = 0.05;
/// Tags a page left without one is asked about, one at a time.
const RESCUE_TRIES: usize = 2;

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
    /// Pages classified at the same time.
    pub concurrency: usize,
}

#[derive(Debug, Clone)]
pub struct VocabTag {
    pub name: String,
    pub description: String,
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
    /// Pages left without a tag that were asked about a tag of their
    /// cluster, and how many of them got it.
    pub rescue_checked: usize,
    pub rescued: usize,
}

const CLUSTER_SYSTEM: &str = "You name a group of pages from a personal archive. The numbered pages were grouped because they are about similar things. Give the one tag for the topic most of them share: the broad topic someone would file them under, like bread, investing, japan, astronomy or machine-learning.

- The tag names a topic, not an aspect of one: astronomy, not astronomy-history; birds, not bird-species.
- Use lowercase kebab-case English. If the word has several common meanings, the most common one keeps the plain word (python, rust, java) and only the others are qualified: python-snake, rust-corrosion, java-island.
- If one of the existing tags names that topic, use its exact name.
- fits lists the numbers of the pages the tag describes.
- If the pages have nothing useful in common, give an empty tag.

Reply with a JSON object: {\"tag\": \"...\", \"description\": \"one line on what the tag covers\", \"fits\": [1, 2, 3]}.";

const CLASSIFY_SYSTEM: &str = "You tag a page of a personal archive with tags from a list. Pick the tags that describe what the page is about.

- Pick 1 to {max} tags, the broadest first.
- If the page is about none of these topics, pick only none.
- A broad tag fits a page about one of its kinds: vehicles fits a page about trucks; sport fits a page about tennis.
- A tag fits only in its own meaning, as its description says: a tag for the programming language never fits a page about the snake.
- Never pick two tags that mean the same thing.
- Use each name exactly as listed, without its description.

Reply with a JSON object: {\"tags\": [\"name\", ...]}.";

const CHECK_SYSTEM: &str = "You check one tag of a personal archive of web pages against one page. Say whether the tag fits the page.

- The tag fits if the page is about its topic, or about one kind or part of it: vehicles fits a page about trucks; sport fits a page about tennis.
- The tag fits only in its own meaning, as its description says: a tag for the programming language never fits a page about the snake.
- It doesn't fit a page that only mentions its topic in passing.

Reply with a JSON object: {\"fits\": true}.";

#[derive(Deserialize)]
struct CheckReply {
    fits: bool,
}

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
/// `word-…`. A vocabulary tag of that form already made for at least half of
/// these pages (a cluster's name) is reused, so one meaning doesn't get two
/// names; one made for other pages (machine-learning, for pages about
/// machine tools) isn't. Otherwise the qualifier is up to two words of the
/// model's name, or of the description if the name has none, leaving out the
/// word itself and words like genus or species that say nothing about the
/// meaning.
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
        .filter(|(shared, _)| *shared > 0 && *shared * 2 >= pages.len())
        .max_by_key(|(shared, _)| *shared);
    if let Some((_, tag)) = existing {
        return Some(tag.clone());
    }
    const SKIP: [&str; 9] = ["the", "a", "an", "of", "and", "in", "for", "its", "to"];
    const GENERIC: [&str; 8] = [
        "genus",
        "species",
        "family",
        "classification",
        "type",
        "kind",
        "category",
        "topic",
    ];
    // The word and its plural, but not other words it begins (golang, google
    // for go; carbon for car).
    let is_word = |w: &str| w == word || [w.strip_suffix('s'), w.strip_suffix("es")].contains(&Some(word));
    let words_of = |text: &str| -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .map(str::to_lowercase)
            .filter(|w| {
                !w.is_empty() && !SKIP.contains(&w.as_str()) && !GENERIC.contains(&w.as_str()) && !is_word(w)
            })
            .take(2)
            .collect()
    };
    let mut qualifier = words_of(name);
    if qualifier.is_empty() {
        qualifier = words_of(description);
    }
    if qualifier.is_empty() {
        return None;
    }
    normalize_name(&format!("{prefix}{}", qualifier.join("-")))
}

fn page_line(page: &AnalyzedPage) -> String {
    if page.topics.is_empty() {
        page.title.clone()
    } else {
        let names: Vec<&str> = page.topics.iter().map(|(name, _)| name.as_str()).collect();
        format!("{} (topics: {})", page.title, names.join(", "))
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
        .filter(|(i, _)| !exclude.contains(&offer.names[*i]) && offer.names[*i] != NONE)
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
        match offer.descriptions[i].trim() {
            "" => user.push_str(&format!("- {}\n", offer.names[i])),
            description => user.push_str(&format!("- {}: {description}\n", offer.names[i])),
        }
    }
    // Offered as a choice, none is picked far more readily than when the
    // prompt only allows an empty reply: a page on a topic the vocabulary
    // lacks is then left untagged instead of given the nearest tag.
    user.push_str(&format!("- {NONE}: the page is about none of these topics\n"));
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

/// Pages of `word` found to be about another of its meanings: its name no
/// longer fits them, the tag `to` of that meaning does (if it has one).
fn move_fit(fit: &mut BTreeMap<String, Vec<usize>>, word: &str, to: Option<&str>, pages: &[usize]) {
    if let Some(list) = fit.get_mut(word) {
        list.retain(|p| !pages.contains(p));
    }
    if let Some(to) = to {
        let list = fit.entry(to.to_string()).or_default();
        for &p in pages {
            if !list.contains(&p) {
                list.push(p);
            }
        }
    }
}

/// The tags to ask a page left without one about, best first: the tag of
/// its cluster, which the naming judged it about, then tags it chose that
/// were dropped for too few pages.
fn rescue_tags(
    page: usize,
    chosen: &[String],
    fit: &BTreeMap<String, Vec<usize>>,
    vocabulary: &BTreeMap<String, VocabTag>,
    locked: &HashSet<String>,
) -> Vec<String> {
    let mut names: Vec<String> = fit
        .iter()
        .filter(|(_, pages)| pages.contains(&page))
        .map(|(name, _)| name.clone())
        .collect();
    for name in chosen {
        if !names.contains(name) {
            names.push(name.clone());
        }
    }
    names.retain(|n| vocabulary.contains_key(n) && !locked.contains(n));
    names.truncate(RESCUE_TRIES);
    names
}

/// Whether the tag fits the page, asked about it alone.
async fn check_tag<R: Reviewer>(
    reviewer: &R,
    page: &AnalyzedPage,
    name: &str,
    description: &str,
) -> Result<bool, String> {
    let tag = match description.trim() {
        "" => name.to_string(),
        description => format!("{name}: {description}"),
    };
    let user = format!(
        "Tag: {tag}\n\nPage: {}\n{}",
        page.title,
        truncate_chars(&page.summary, SUMMARY_CHARS)
    );
    // A field for what the page is about, before the verdict, made no
    // difference on 100 hard cases.
    let schema = json!({
        "type": "object", "additionalProperties": false, "required": ["fits"],
        "properties": {"fits": {"type": "boolean"}}
    });
    let reply = reviewer
        .ask_json(CHECK_SYSTEM, &user, "tag_check", schema)
        .await
        .map_err(|err| format!("{err:#}"))?;
    serde_json::from_value::<CheckReply>(reply)
        .map(|r| r.fits)
        .map_err(|_| "a tag check was unusable".to_string())
}

/// What the model said about a word with several meanings.
struct Split {
    /// What the word names in its most common meaning, the one keeping it.
    main_description: String,
    /// Each other meaning: its qualified tag, description and pages.
    others: Vec<(String, String, Vec<usize>)>,
}

/// Asks whether `word`, on the pages `list` (indexes into `pages`), names
/// unrelated things. `None` when it doesn't: one meaning, all pages in one
/// cluster, no other meaning that could be named, or a failed request
/// (recorded in `out`).
///
/// A one-meaning verdict holds for the rest of the review (in `same`) until
/// the word has twice the pages. If `remember`, it is also saved with the
/// pages the model was shown, and a later review reuses it only when it
/// would show the same pages in the same groups: at temperature 0 asking
/// again would get the same answer. So a review of an archive that grew
/// checks the word as a review of all its pages from scratch would.
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
    same: &mut BTreeMap<String, usize>,
    out: &mut Retag,
) -> Result<Option<Split>> {
    if same.get(word).is_some_and(|&n| list.len() < n * 2) {
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
    // What the model is asked: the pages shown, group by group.
    let asked = shown
        .iter()
        .map(|g| g.iter().map(|p| p.id.to_string()).collect::<Vec<_>>().join(","))
        .collect::<Vec<_>>()
        .join("|");
    let key = format!("meanings:{word}");
    let verdict = format!("same:{asked}");
    if db.tag_decision(&key)?.as_deref() == Some(verdict.as_str()) {
        if remember {
            same.insert(word.to_string(), list.len());
        }
        return Ok(None);
    }
    out.meaning_checks += 1;
    // A word checked again after classification is listed once.
    if !out.checked.iter().any(|(w, _)| w == word) {
        out.checked.push((
            word.to_string(),
            shown
                .iter()
                .filter_map(|g| g.first().map(|p| p.title.clone()))
                .collect(),
        ));
    }
    let things = match meanings::check(reviewer, word, &shown).await {
        Ok(meanings::Verdict::Same) => {
            if remember {
                same.insert(word.to_string(), list.len());
                db.set_tag_decision(&key, &verdict, "llm")?;
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

/// What naming the clusters has done so far, besides the vocabulary.
#[derive(Default)]
struct Naming {
    /// Pages a cluster's name fits, and those past the sample the model saw.
    named: HashSet<usize>,
    /// For each tag a cluster described, how many pages that cluster's name
    /// fit.
    described: HashMap<String, usize>,
    /// For each name, the pages of its clusters that it fit, and those past
    /// the sample the model saw: a page left without a tag is asked about it.
    fit: BTreeMap<String, Vec<usize>>,
}

/// Asks the model to name `cluster` (indexes into `pages`) and adds the name
/// to the vocabulary, with the pages it fits as members.
#[allow(clippy::too_many_arguments)]
async fn name_cluster<R: Reviewer>(
    reviewer: &R,
    cluster: &[usize],
    pages: &[AnalyzedPage],
    schema: &serde_json::Value,
    settings: &Settings,
    vocabulary: &mut BTreeMap<String, VocabTag>,
    members: &mut BTreeMap<String, Vec<usize>>,
    naming: &mut Naming,
    out: &mut Retag,
) {
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
        .ask_json(CLUSTER_SYSTEM, &user, "cluster_tag", schema.clone())
        .await
    {
        Ok(reply) => reply,
        Err(err) => {
            out.failed_requests.push(format!("{err:#}"));
            return;
        }
    };
    let Ok(reply) = serde_json::from_value::<ClusterReply>(reply) else {
        out.failed_requests.push("a cluster name was unusable".into());
        return;
    };
    let Some(name) = normalize_name(&reply.tag) else {
        return;
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
        return;
    }
    let fits = fitting.len();
    naming.named.extend(cluster.iter().skip(sample.len()));
    let fit = naming.fit.entry(name.clone()).or_default();
    for &i in fitting.iter().chain(cluster.iter().skip(sample.len())) {
        if !fit.contains(&i) {
            fit.push(i);
        }
    }
    let list = members.entry(name.clone()).or_default();
    for i in fitting {
        naming.named.insert(i);
        if !list.contains(&i) {
            list.push(i);
        }
    }
    // A cluster's description was written for its pages, so it replaces the
    // one a topic got from the first page it was used on. When two clusters
    // get the same name, the one the name fits better describes it: the
    // classifier reads the description, and one written for a few stray
    // pages (ancient-rome: "landmarks and volcanic activity in sicily")
    // keeps the tag off pages it should have.
    let description = reply.description.trim().to_string();
    if description.is_empty() || naming.described.get(&name).is_some_and(|&n| n >= fits) {
        vocabulary.entry(name.clone()).or_insert(VocabTag {
            name,
            description: String::new(),
            pages: 0,
        });
        return;
    }
    naming.described.insert(name.clone(), fits);
    match vocabulary.get_mut(&name) {
        Some(tag) => tag.description = description,
        None => {
            vocabulary.insert(
                name.clone(),
                VocabTag {
                    name,
                    description,
                    pages: 0,
                },
            );
        }
    }
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
    // The review reads what each page had when a review first saw it, not
    // the tags an earlier review gave it: reviewing again gives the same tags.
    db.freeze_review_topics()?;
    let pages = db.analyzed_pages()?;
    if pages.is_empty() {
        return Ok(out);
    }
    let texts: Vec<String> = pages
        .iter()
        .map(|p| {
            let topics: Vec<&str> = p.topics.iter().map(|(name, _)| name.as_str()).collect();
            format!(
                "Title: {}\nSummary: {}\nTopics: {}",
                p.title,
                p.summary,
                topics.join("; ")
            )
        })
        .collect();
    let page_vectors = embed_all(embedder, &texts).await?;

    // Topics several pages already share, with the description each had.
    let mut descriptions: HashMap<&str, &str> = HashMap::new();
    for page in &pages {
        for (name, description) in &page.topics {
            let known = descriptions.entry(name.as_str()).or_default();
            if known.is_empty() {
                *known = description.as_str();
            }
        }
    }
    let locked: HashSet<String> = db
        .topic_tags()?
        .into_iter()
        .filter(|t| t.locked)
        .map(|t| t.name)
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
        for (tag, _) in &page.topics {
            *topic_counts.entry(tag.as_str()).or_default() += 1;
        }
        if let Some((first, _)) = page.topics.first() {
            first_tags.insert(first.as_str());
        }
    }
    let mut vocabulary: BTreeMap<String, VocabTag> = BTreeMap::new();
    // Pages each vocabulary tag was made for, by index.
    let mut members: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, page) in pages.iter().enumerate() {
        for (tag, _) in &page.topics {
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
                description: descriptions
                    .get(name.as_str())
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                pages: 0,
            },
        );
    }

    // Which pages are alike, for the meanings check below.
    let all_clusters = agglomerate(&page_vectors, settings.cluster_similarity);
    let mut cluster_of = vec![0; pages.len()];
    for (c, list) in all_clusters.iter().enumerate() {
        for &i in list {
            cluster_of[i] = c;
        }
    }
    note(&format!(
        "Building the vocabulary: {} shared topics, then names for clusters of pages.",
        vocabulary.len()
    ));
    // The model writes the fields in this order: what the pages share, which
    // pages, then the name.
    let cluster_schema = json!({
        "type": "object", "additionalProperties": false, "required": ["description", "fits", "tag"],
        "properties": {
            "description": {"type": "string"},
            "fits": {"type": "array", "items": {"type": "integer"}},
            "tag": {"type": "string"}
        }
    });
    // A name for each cluster big enough to deserve a tag, in two passes.
    // Clusters of very alike pages come first: at the looser similarity two
    // neighbouring topics (dogs and horses, fruit and houseplants) fall into
    // one cluster, which gets named for one of them. The pages no name fits
    // yet are then clustered again at the looser similarity, which finds the
    // topics whose pages are less alike (investing, history).
    let mut naming = Naming::default();
    for pass in 0..2 {
        let clusters: Vec<Vec<usize>> = if pass == 0 {
            agglomerate(&page_vectors, settings.cluster_similarity + TIGHT_STEP)
        } else {
            let pool: Vec<usize> = (0..pages.len()).filter(|i| !naming.named.contains(i)).collect();
            let vectors: Vec<Vec<f32>> = pool.iter().map(|&i| page_vectors[i].clone()).collect();
            agglomerate(&vectors, settings.cluster_similarity)
                .into_iter()
                .map(|c| c.into_iter().map(|j| pool[j]).collect())
                .collect()
        };
        for cluster in clusters.iter().filter(|c| c.len() >= settings.min_support) {
            name_cluster(
                reviewer,
                cluster,
                &pages,
                &cluster_schema,
                settings,
                &mut vocabulary,
                &mut members,
                &mut naming,
                &mut out,
            )
            .await;
        }
    }
    if vocabulary.is_empty() {
        return Ok(out);
    }

    // A one-word topic whose pages fall into different clusters may be one
    // word for unrelated things (python the language and the snake): the
    // model says whether it is, and each meaning gets a qualified tag. This
    // includes a name the model gave two unrelated clusters (amazon for the
    // company and for the rainforest).
    let mut same: BTreeMap<String, usize> = BTreeMap::new();
    let words: Vec<String> = vocabulary
        .values()
        .filter(|t| !t.name.contains('-'))
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
            &mut same,
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
                pages: 0,
            });
            move_fit(&mut naming.fit, &word, Some(&name), &thing_pages);
            // So a later check of the word reuses this name for these pages.
            members.entry(name.clone()).or_default().extend(thing_pages);
            names.push(name);
        }
        // The word now names that meaning: its description says so, so a page
        // about another meaning isn't given it.
        if let Some(tag) = vocabulary.get_mut(&word)
            && !split.main_description.is_empty()
        {
            tag.description = split.main_description.clone();
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
    // None for a page whose request failed: it keeps its tags. The pages
    // are independent of each other, so several are asked at once.
    let replies: Vec<Result<Vec<String>, String>> = stream::iter(pages.iter().zip(&page_vectors))
        .map(|(page, vector)| classify(reviewer, &offer, page, vector, settings, &[]))
        .buffered(settings.concurrency.max(1))
        .collect()
        .await;
    let mut chosen: Vec<Option<Vec<String>>> = Vec::with_capacity(pages.len());
    for reply in replies {
        match reply {
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
            &mut same,
            &mut out,
        )
        .await?
        else {
            continue;
        };
        let mut names = vec![word.clone()];
        for (name, _, thing_pages) in split.others {
            let known = vocabulary.contains_key(&name);
            move_fit(
                &mut naming.fit,
                &word,
                known.then_some(name.as_str()),
                &thing_pages,
            );
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
            // A meaning without a tag of its own made none: its pages are
            // tagged again.
            if known {
                names.push(name);
            }
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
    // sneakers) stays on a page nothing else fits. When the model finds
    // nothing, or its request fails, the page keeps the tags it has left,
    // and gets the word back only if it has none.
    for (i, (before, exclude)) in retry {
        match classify(reviewer, &offer, &pages[i], &page_vectors[i], settings, &exclude).await {
            Ok(tags) if !tags.is_empty() => chosen[i] = Some(tags),
            result => {
                if let Err(err) = result {
                    out.failed_requests.push(err);
                }
                if chosen[i].as_ref().is_some_and(Vec::is_empty) {
                    chosen[i] = Some(before);
                }
            }
        }
    }

    // Tags too few pages chose are dropped.
    let mut counts: HashMap<String, usize> = HashMap::new();
    for tags in chosen.iter().flatten() {
        for t in tags {
            *counts.entry(t.clone()).or_default() += 1;
        }
    }
    let mut kept: HashSet<String> = counts
        .iter()
        .filter(|(_, n)| **n >= settings.min_support)
        .map(|(t, _)| t.to_string())
        .collect();
    out.dropped = names.iter().filter(|n| !kept.contains(*n)).cloned().collect();
    // A page all of whose tags were dropped is tagged again from the tags
    // kept: a narrow tag (string-instrument on a page about the violin) can
    // be dropped while a broader one (music) fits the page as well. Its
    // tags can only gain pages, so no kept tag is dropped by this.
    let orphans: Vec<usize> = chosen
        .iter()
        .enumerate()
        .filter(|(_, tags)| {
            tags.as_ref()
                .is_some_and(|t| !t.is_empty() && t.iter().all(|t| !kept.contains(t)))
        })
        .map(|(i, _)| i)
        .collect();
    if !kept.is_empty() {
        for i in orphans {
            match classify(
                reviewer,
                &offer,
                &pages[i],
                &page_vectors[i],
                settings,
                &out.dropped,
            )
            .await
            {
                Ok(tags) if !tags.is_empty() => {
                    for t in &tags {
                        *counts.entry(t.clone()).or_default() += 1;
                    }
                    chosen[i] = Some(tags);
                }
                Ok(_) => {}
                Err(err) => out.failed_requests.push(err),
            }
        }
    }

    // A page still without a tag is asked, yes or no, about a tag its cluster's
    // name fit, or one it chose that was dropped. Picking from a list, the
    // model answers none even when that tag comes first; when several pages
    // of a topic do, its tag falls under min_support and the topic vanishes.
    // One tag at a time is an easier question. A dropped tag comes back if
    // enough pages say yes.
    let candidates: Vec<(usize, Vec<String>)> = chosen
        .iter()
        .enumerate()
        .filter_map(|(i, tags)| {
            let tags = tags.as_ref()?;
            if tags.iter().any(|t| kept.contains(t)) || pages[i].tags.iter().any(|t| locked.contains(t)) {
                return None;
            }
            let names = rescue_tags(i, tags, &naming.fit, &vocabulary, &locked);
            (!names.is_empty()).then_some((i, names))
        })
        .collect();
    // A dropped tag too few of these pages could say yes to can't come back.
    let mut could_say_yes: HashMap<&str, usize> = HashMap::new();
    for (_, names) in &candidates {
        for name in names {
            *could_say_yes.entry(name.as_str()).or_default() += 1;
        }
    }
    let worth_asking: HashSet<String> = could_say_yes
        .into_iter()
        .filter(|(name, n)| kept.contains(*name) || *n >= settings.min_support)
        .map(|(name, _)| name.to_string())
        .collect();
    let candidates: Vec<(usize, Vec<String>)> = candidates
        .into_iter()
        .map(|(i, names)| {
            (
                i,
                names.into_iter().filter(|n| worth_asking.contains(n)).collect(),
            )
        })
        .filter(|(_, names): &(usize, Vec<String>)| !names.is_empty())
        .collect();
    out.rescue_checked = candidates.len();
    let answers: Vec<(usize, Option<String>, Vec<String>)> = stream::iter(candidates)
        .map(|(i, names)| {
            let (page, vocabulary) = (&pages[i], &vocabulary);
            async move {
                let mut errors = Vec::new();
                for name in names {
                    match check_tag(reviewer, page, &name, &vocabulary[&name].description).await {
                        Ok(true) => return (i, Some(name), errors),
                        Ok(false) => {}
                        Err(err) => errors.push(err),
                    }
                }
                (i, None, errors)
            }
        })
        .buffered(settings.concurrency.max(1))
        .collect()
        .await;
    let mut said_yes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, name, errors) in answers {
        out.failed_requests.extend(errors);
        if let Some(name) = name {
            said_yes.entry(name).or_default().push(i);
        }
    }
    for (name, list) in said_yes {
        if !kept.contains(&name) && list.len() < settings.min_support {
            continue;
        }
        kept.insert(name.clone());
        for i in list {
            chosen[i] = Some(vec![name.clone()]);
            out.rescued += 1;
        }
    }
    if out.rescue_checked > 0 {
        note(&format!(
            "Asked {} pages left without a tag about their cluster's tag: {} got it.",
            out.rescue_checked, out.rescued
        ));
    }
    out.dropped = names.iter().filter(|n| !kept.contains(*n)).cloned().collect();
    // Pages per tag, as assigned.
    let mut counts: HashMap<String, usize> = HashMap::new();
    for tags in chosen.iter().flatten() {
        for t in tags.iter().filter(|t| kept.contains(*t)) {
            *counts.entry(t.clone()).or_default() += 1;
        }
    }

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
            // The snake page lost python-snake, used on it alone, and is
            // tagged again from coffee and python.
            json!({"tags": ["none"]}),
        ]);
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 2,
            candidates: 8,
            max_tags: 3,
            concurrency: 1,
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
        // The one-meaning verdict is remembered with the pages shown, one
        // group each, but not python's second check, which only saw the
        // language.
        assert_eq!(
            db.tag_decision("meanings:coffee").unwrap().as_deref(),
            Some("same:1|2|3")
        );
        assert_eq!(db.tag_decision("meanings:python").unwrap(), None);
    }

    #[tokio::test]
    async fn a_one_meaning_verdict_is_reused_only_for_the_same_pages() {
        use crate::reconcile::tests::{FakeReviewer, archive, no_names};
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 2,
            candidates: 8,
            max_tags: 3,
            concurrency: 1,
        };
        // Pages unlike each other: each is its own group in the check.
        let pages = [
            (&["coffee"][..], [1.0, 0.0, 0.0]),
            (&["coffee"][..], [0.0, 1.0, 0.0]),
            (&["coffee"][..], [0.0, 0.0, 1.0]),
        ];
        let same = || json!({"groups": [], "same_meaning": true});
        let coffee = || json!({"tags": ["coffee"]});
        let db = archive(&pages);
        let mut replies = vec![same()];
        replies.extend((0..6).map(|_| coffee()));
        let reviewer = FakeReviewer::new(replies);
        let first = retag(&db, &reviewer, &no_names(), &settings, |_| {})
            .await
            .unwrap();
        // The same pages: the same question, so the verdict holds.
        let second = retag(&db, &reviewer, &no_names(), &settings, |_| {})
            .await
            .unwrap();
        assert_eq!((first.meaning_checks, second.meaning_checks), (1, 0));

        // With a fourth page the question changes, so it is asked again, as
        // a review of the four pages from scratch would.
        let grown = archive(&[pages[0], pages[1], pages[2], (&["coffee"][..], [0.5, 0.5, 0.5])]);
        let verdict = db.tag_decision("meanings:coffee").unwrap().unwrap();
        grown
            .set_tag_decision("meanings:coffee", &verdict, "llm")
            .unwrap();
        let mut replies = vec![same()];
        replies.extend((0..4).map(|_| coffee()));
        let out = retag(
            &grown,
            &FakeReviewer::new(replies),
            &no_names(),
            &settings,
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(out.meaning_checks, 1);
        assert_eq!(
            grown.tag_decision("meanings:coffee").unwrap().as_deref(),
            Some("same:1|2|3|4")
        );
    }

    /// Embeds a page by its first topic, from a fixed table; other texts get
    /// a vector of their own, unlike any other.
    struct TopicEmbedder(Vec<(&'static str, [f32; 2])>);

    impl Embedder for TopicEmbedder {
        fn model(&self) -> &str {
            "m"
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    let mut out = vec![0.0; 2 + texts.len()];
                    let first = text
                        .lines()
                        .find_map(|l| l.strip_prefix("Topics: "))
                        .and_then(|t| t.split("; ").next());
                    match self.0.iter().find(|(topic, _)| Some(*topic) == first) {
                        Some((_, v)) => out[..2].copy_from_slice(v),
                        None => out[2 + i] = 1.0,
                    }
                    out
                })
                .collect())
        }
    }

    /// Three dog pages and three horse pages, each with a topic of its own.
    fn dogs_and_horses() -> Db {
        use crate::reconcile::tests::archive;
        archive(&[
            (&["labrador"], [0.0; 3]),
            (&["poodle"], [0.0; 3]),
            (&["beagle"], [0.0; 3]),
            (&["mustang"], [0.0; 3]),
            (&["arabian"], [0.0; 3]),
            (&["pony"], [0.0; 3]),
        ])
    }

    fn tags_of(out: &Retag) -> Vec<Vec<&str>> {
        out.assignments
            .iter()
            .map(|(_, t)| t.iter().map(|(n, _)| n.as_str()).collect())
            .collect()
    }

    #[tokio::test]
    async fn neighbouring_topics_get_a_name_each() {
        use crate::reconcile::tests::FakeReviewer;
        // Dogs and horses are 0.82 alike: one cluster at 0.80, two at 0.85.
        let embedder = TopicEmbedder(
            [
                ("labrador", 1),
                ("poodle", 1),
                ("beagle", 1),
                ("mustang", -1),
                ("arabian", -1),
                ("pony", -1),
            ]
            .map(|(topic, sign)| (topic, [1.0, sign as f32 * 0.31]))
            .to_vec(),
        );
        let reviewer = FakeReviewer::new(vec![
            json!({"description": "Dogs.", "fits": [1, 2, 3], "tag": "dogs"}),
            json!({"description": "Horses.", "fits": [1, 2, 3], "tag": "horses"}),
            json!({"tags": ["dogs"]}),
            json!({"tags": ["dogs"]}),
            json!({"tags": ["dogs"]}),
            json!({"tags": ["horses"]}),
            json!({"tags": ["horses"]}),
            json!({"tags": ["horses"]}),
        ]);
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 3,
            candidates: 8,
            max_tags: 3,
            concurrency: 1,
        };
        let out = retag(&dogs_and_horses(), &reviewer, &embedder, &settings, |_| {})
            .await
            .unwrap();
        let first = &reviewer.prompts.borrow()[0];
        assert!(
            first.contains("labrador") && !first.contains("mustang"),
            "{first}"
        );
        assert_eq!(
            tags_of(&out),
            [
                vec!["dogs"],
                vec!["dogs"],
                vec!["dogs"],
                vec!["horses"],
                vec!["horses"],
                vec!["horses"]
            ]
        );
    }

    /// Dog pages and horse pages, 0.82 alike: a cluster each at 0.85.
    fn dogs_apart_from_horses() -> TopicEmbedder {
        TopicEmbedder(
            [
                ("labrador", 1),
                ("poodle", 1),
                ("beagle", 1),
                ("mustang", -1),
                ("arabian", -1),
                ("pony", -1),
            ]
            .map(|(topic, sign)| (topic, [1.0, sign as f32 * 0.31]))
            .to_vec(),
        )
    }

    #[tokio::test]
    async fn a_topic_whose_pages_answer_none_comes_back_if_they_say_yes_to_its_tag() {
        use crate::reconcile::tests::FakeReviewer;
        let fits = |yes: bool| json!({"fits": yes});
        let reviewer = FakeReviewer::new(vec![
            json!({"description": "Dogs.", "fits": [1, 2, 3], "tag": "dogs"}),
            json!({"description": "Horses.", "fits": [1, 2, 3], "tag": "horses"}),
            json!({"tags": ["dogs"]}),
            json!({"tags": ["dogs"]}),
            json!({"tags": ["dogs"]}),
            // Two horse pages answer none, so horses has one page and is
            // dropped; the page that chose it finds nothing else.
            json!({"tags": ["horses"]}),
            json!({"tags": ["none"]}),
            json!({"tags": ["none"]}),
            json!({"tags": ["none"]}),
            // Each horse page is asked about the horses tag alone.
            fits(true),
            fits(true),
            fits(true),
        ]);
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 3,
            candidates: 8,
            max_tags: 3,
            concurrency: 1,
        };
        let out = retag(
            &dogs_and_horses(),
            &reviewer,
            &dogs_apart_from_horses(),
            &settings,
            |_| {},
        )
        .await
        .unwrap();
        let last = reviewer.prompts.borrow().last().cloned().unwrap();
        assert!(last.starts_with("Tag: horses: Horses.\n\nPage: "), "{last}");
        assert_eq!(
            tags_of(&out),
            [
                vec!["dogs"],
                vec!["dogs"],
                vec!["dogs"],
                vec!["horses"],
                vec!["horses"],
                vec!["horses"]
            ]
        );
        assert_eq!((out.rescue_checked, out.rescued), (3, 3));
        assert!(out.dropped.is_empty());
        assert_eq!(out.untagged, 0);
    }

    #[tokio::test]
    async fn a_page_that_answers_none_gets_its_cluster_tag_if_it_fits_and_too_few_yeses_keep_none() {
        use crate::reconcile::tests::FakeReviewer;
        let fits = |yes: bool| json!({"fits": yes});
        let reviewer = FakeReviewer::new(vec![
            json!({"description": "Dogs.", "fits": [1, 2, 3], "tag": "dogs"}),
            json!({"description": "Horses.", "fits": [1, 2, 3], "tag": "horses"}),
            // dogs keeps 2 pages, enough; horses has 1 and is dropped.
            json!({"tags": ["dogs"]}),
            json!({"tags": ["dogs"]}),
            json!({"tags": ["none"]}),
            json!({"tags": ["horses"]}),
            json!({"tags": ["none"]}),
            json!({"tags": ["none"]}),
            // The page that chose horses finds nothing else.
            json!({"tags": ["none"]}),
            // The dog page gets dogs; of the horse pages only one says yes,
            // too few to bring horses back.
            fits(true),
            fits(true),
            fits(false),
            fits(false),
        ]);
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 2,
            candidates: 8,
            max_tags: 3,
            concurrency: 1,
        };
        let out = retag(
            &dogs_and_horses(),
            &reviewer,
            &dogs_apart_from_horses(),
            &settings,
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(
            tags_of(&out),
            [vec!["dogs"], vec!["dogs"], vec!["dogs"], vec![], vec![], vec![]]
        );
        assert_eq!((out.rescue_checked, out.rescued), (4, 1));
        assert_eq!(out.dropped, ["horses"]);
        assert_eq!(out.untagged, 3);
    }

    #[tokio::test]
    async fn pages_a_cluster_name_leaves_out_are_named_again() {
        use crate::reconcile::tests::FakeReviewer;
        // All six pages are alike: one cluster, which the model names for
        // its dog pages only.
        let embedder = TopicEmbedder(
            ["labrador", "poodle", "beagle", "mustang", "arabian", "pony"]
                .map(|topic| (topic, [1.0, 0.0]))
                .to_vec(),
        );
        let reviewer = FakeReviewer::new(vec![
            json!({"description": "Dogs.", "fits": [1, 2, 3], "tag": "dogs"}),
            json!({"description": "Horses.", "fits": [1, 2, 3], "tag": "horses"}),
            json!({"tags": ["dogs"]}),
            json!({"tags": ["dogs"]}),
            json!({"tags": ["dogs"]}),
            json!({"tags": ["horses"]}),
            json!({"tags": ["horses"]}),
            json!({"tags": ["horses"]}),
        ]);
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 3,
            candidates: 8,
            max_tags: 3,
            concurrency: 1,
        };
        let out = retag(&dogs_and_horses(), &reviewer, &embedder, &settings, |_| {})
            .await
            .unwrap();
        // The second request shows only the pages left out, with dogs as an
        // existing tag.
        let second = &reviewer.prompts.borrow()[1];
        assert!(second.starts_with("Existing tags: dogs\n"), "{second}");
        assert!(
            second.contains("mustang") && !second.contains("labrador"),
            "{second}"
        );
        let names: Vec<&str> = out.vocabulary.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["dogs", "horses"]);
    }

    #[tokio::test]
    async fn a_page_about_none_of_the_tags_is_left_untagged() {
        use crate::reconcile::tests::FakeReviewer;
        // The dog pages are alike; the horse pages are unlike anything.
        let embedder = TopicEmbedder(
            ["labrador", "poodle", "beagle"]
                .map(|topic| (topic, [1.0, 0.0]))
                .to_vec(),
        );
        let reviewer = FakeReviewer::new(vec![
            json!({"description": "Dogs.", "fits": [1, 2, 3], "tag": "dogs"}),
            json!({"tags": ["dogs"]}),
            json!({"tags": ["dogs"]}),
            json!({"tags": ["dogs"]}),
            json!({"tags": ["none"]}),
            json!({"tags": ["none"]}),
            json!({"tags": ["none"]}),
        ]);
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 3,
            candidates: 8,
            max_tags: 3,
            concurrency: 1,
        };
        let out = retag(&dogs_and_horses(), &reviewer, &embedder, &settings, |_| {})
            .await
            .unwrap();
        let last = reviewer.prompts.borrow().last().cloned().unwrap();
        assert!(
            last.ends_with("- dogs: Dogs.\n- none: the page is about none of these topics\n"),
            "{last}"
        );
        assert_eq!(
            tags_of(&out),
            [vec!["dogs"], vec!["dogs"], vec!["dogs"], vec![], vec![], vec![]]
        );
        assert_eq!(out.untagged, 3);
    }

    /// Four pages about one thing and three about another, unlike each
    /// other, each page with a topic of its own.
    fn two_groups() -> (Db, TopicEmbedder) {
        use crate::reconcile::tests::archive;
        let db = archive(&[
            (&["a1"], [0.0; 3]),
            (&["a2"], [0.0; 3]),
            (&["a3"], [0.0; 3]),
            (&["a4"], [0.0; 3]),
            (&["b1"], [0.0; 3]),
            (&["b2"], [0.0; 3]),
            (&["b3"], [0.0; 3]),
        ]);
        let embedder = TopicEmbedder(
            ["a1", "a2", "a3", "a4", "b1", "b2", "b3"]
                .map(|topic| {
                    (
                        topic,
                        if topic.starts_with('a') {
                            [1.0, 0.0]
                        } else {
                            [0.0, 1.0]
                        },
                    )
                })
                .to_vec(),
        );
        (db, embedder)
    }

    #[tokio::test]
    async fn a_name_two_clusters_get_keeps_the_description_it_fits_best() {
        use crate::reconcile::tests::FakeReviewer;
        let (db, embedder) = two_groups();
        let mut replies = vec![
            json!({"description": "Ancient Rome.", "fits": [1, 2, 3, 4], "tag": "ancient-rome"}),
            json!({"description": "Roman landmarks and volcanoes.", "fits": [1, 2, 3], "tag": "ancient-rome"}),
        ];
        replies.extend((0..7).map(|_| json!({"tags": ["ancient-rome"]})));
        let reviewer = FakeReviewer::new(replies);
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 3,
            candidates: 8,
            max_tags: 3,
            concurrency: 1,
        };
        let out = retag(&db, &reviewer, &embedder, &settings, |_| {}).await.unwrap();
        // The first name fits four pages, the second three.
        assert_eq!(out.vocabulary[0].description, "Ancient Rome.");
    }

    #[tokio::test]
    async fn a_word_two_unrelated_clusters_are_named_is_split_by_meaning() {
        use crate::reconcile::tests::FakeReviewer;
        let (db, embedder) = two_groups();
        let mut replies = vec![
            json!({"description": "Amazon, the company.", "fits": [1, 2, 3, 4], "tag": "amazon"}),
            json!({"description": "The Amazon rainforest.", "fits": [1, 2, 3], "tag": "amazon"}),
            json!({"groups": [], "same_meaning": false}),
            json!({"referent": "the company"}),
            json!({"referent": "the rainforest"}),
            json!({"groups": [
                {"group": 1, "tag": "amazon", "description": "Amazon, the company."},
                {"group": 2, "tag": "amazon-rainforest", "description": "The Amazon rainforest."}]}),
        ];
        replies.extend((0..4).map(|_| json!({"tags": ["amazon"]})));
        replies.extend((0..3).map(|_| json!({"tags": ["amazon-rainforest"]})));
        let reviewer = FakeReviewer::new(replies);
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 3,
            candidates: 8,
            max_tags: 3,
            concurrency: 1,
        };
        let out = retag(&db, &reviewer, &embedder, &settings, |_| {}).await.unwrap();
        assert_eq!(out.meaning_checks, 1);
        let mut expected = vec![vec!["amazon"]; 4];
        expected.extend(vec![vec!["amazon-rainforest"]; 3]);
        assert_eq!(tags_of(&out), expected);
    }

    #[tokio::test]
    async fn a_page_that_loses_a_wrong_meaning_keeps_its_other_tags() {
        use crate::reconcile::tests::FakeReviewer;
        let (db, embedder) = two_groups();
        let mut replies = vec![
            json!({"description": "Rust, the programming language.", "fits": [1, 2, 3, 4], "tag": "rust"}),
            json!({"description": "Chemistry.", "fits": [1, 2, 3], "tag": "chemistry"}),
        ];
        // rust is given to the chemistry pages too.
        replies.extend((0..4).map(|_| json!({"tags": ["rust"]})));
        replies.extend([
            json!({"tags": ["rust", "chemistry"]}),
            json!({"tags": ["rust", "chemistry"]}),
            json!({"tags": ["rust"]}),
            // rust, given to pages of both clusters, has two meanings; the
            // other one has no tag in the vocabulary.
            json!({"groups": [], "same_meaning": false}),
            json!({"referent": "the programming language"}),
            json!({"referent": "iron oxide"}),
            json!({"groups": [
                {"group": 1, "tag": "rust", "description": "Rust, the language."},
                {"group": 2, "tag": "rust-corrosion", "description": "Rust on iron."}]}),
            // The chemistry pages are tagged again without rust: nothing
            // fits one, the requests for the other two fail.
            json!({"tags": ["none"]}),
            json!({"tags": 5}),
            json!({"tags": 5}),
        ]);
        let reviewer = FakeReviewer::new(replies);
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 2,
            candidates: 8,
            max_tags: 3,
            concurrency: 1,
        };
        let out = retag(&db, &reviewer, &embedder, &settings, |_| {}).await.unwrap();
        // Two pages keep chemistry; the last, with nothing else, keeps rust.
        let mut expected = vec![vec!["rust"]; 4];
        expected.extend([vec!["chemistry"], vec!["chemistry"], vec!["rust"]]);
        assert_eq!(tags_of(&out), expected);
        assert_eq!(out.failed_requests.len(), 2);
        // rust-corrosion was never made, so it isn't listed.
        assert_eq!(out.ambiguous, [("rust".to_string(), vec!["rust".to_string()])]);
        assert_eq!(out.checked.len(), 1);
    }

    #[tokio::test]
    async fn a_page_whose_tags_are_all_dropped_is_tagged_again() {
        use crate::reconcile::tests::FakeReviewer;
        let (db, embedder) = two_groups();
        let mut replies = vec![
            json!({"description": "Music.", "fits": [1, 2, 3, 4], "tag": "music"}),
            json!({"description": "String instruments.", "fits": [1, 2, 3], "tag": "string-instrument"}),
        ];
        replies.extend((0..4).map(|_| json!({"tags": ["music"]})));
        replies.extend([
            json!({"tags": ["string-instrument"]}),
            json!({"tags": ["string-instrument"]}),
            json!({"tags": ["music"]}),
            // music, given to pages of both clusters, has one meaning.
            json!({"groups": [], "same_meaning": true}),
            // string-instrument, on two pages, is dropped: they are tagged
            // again without it.
            json!({"tags": ["music"]}),
            json!({"tags": ["music"]}),
        ]);
        let reviewer = FakeReviewer::new(replies);
        let settings = Settings {
            cluster_similarity: 0.8,
            min_support: 3,
            candidates: 8,
            max_tags: 3,
            concurrency: 1,
        };
        let out = retag(&db, &reviewer, &embedder, &settings, |_| {}).await.unwrap();
        let last = reviewer.prompts.borrow().last().cloned().unwrap();
        assert!(!last.contains("string-instrument"), "{last}");
        assert_eq!(tags_of(&out), vec![vec!["music"]; 7]);
        assert_eq!(out.dropped, ["string-instrument"]);
        assert_eq!(out.untagged, 0);
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
            concurrency: 1,
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
            concurrency: 1,
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
            ("machine-learning".to_string(), vec![20, 21, 22]),
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
            name("java", "java-island", "the island", &[7, 9]).as_deref(),
            Some("java-indonesia")
        );
        // One made mostly for other pages isn't.
        assert_eq!(
            name("machine", "machine-tools", "machine tools", &[22, 30, 31]).as_deref(),
            Some("machine-tools")
        );
        // Only the word and its plural are left out, not words it begins.
        assert_eq!(
            name("go", "go-golang", "the language from google", &[3]).as_deref(),
            Some("go-golang")
        );
        assert_eq!(
            name("go", "go", "a board game", &[3]).as_deref(),
            Some("go-board-game")
        );
        assert_eq!(
            name("car", "cars", "carbon fibre bodies", &[3]).as_deref(),
            Some("car-carbon-fibre")
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
        // Generic words and repeats of the word say nothing about the meaning.
        assert_eq!(
            name("python", "python-snake-genus", "a genus of snakes", &[3]).as_deref(),
            Some("python-snake")
        );
        assert_eq!(
            name("botany", "botany-botany", "the coffee plant", &[3]).as_deref(),
            Some("botany-coffee-plant")
        );
        assert_eq!(
            name("taxonomy", "taxonomy-fish-species-classification", "", &[3]).as_deref(),
            Some("taxonomy-fish")
        );
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

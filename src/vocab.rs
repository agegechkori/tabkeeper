//! Building the tag vocabulary from the whole archive, then tagging every page
//! from it. Pages are clustered by what they're about, the model names each
//! cluster, topics already used on several pages join the vocabulary, and each
//! page then gets its tags from the vocabulary tags nearest to it.

use std::collections::{BTreeMap, HashMap, HashSet};

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
    /// Sizes of the clusters with at least `min_support` pages.
    pub clusters: Vec<usize>,
    pub vocabulary: Vec<VocabTag>,
    /// Tags offered or chosen but used on too few pages in the end.
    pub dropped: Vec<String>,
    /// (page, its new tags as (name, description)).
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

const CLUSTER_SYSTEM: &str = "You build the tag vocabulary of a personal archive of web pages. The numbered pages were grouped because they are about similar things. Give the one tag that best describes what most of them have in common: broad enough to fit several of them, specific enough to find them by, like bread, investing, japan, classical-music or rust-programming.

- Use lowercase kebab-case English. If the word has several common meanings, qualify it: python-snake, rust-corrosion, java-island.
- If one of the existing tags fits, use its exact name.
- fits lists the numbers of the pages the tag describes.
- If the pages have nothing useful in common, give an empty tag.

Reply with a JSON object: {\"tag\": \"...\", \"description\": \"one line on what the tag covers\", \"fits\": [1, 2, 3]}.";

const CLASSIFY_SYSTEM: &str = "You tag a page of a personal archive with tags from a fixed list. Choose the tags that describe what the page is about: 1 to {max}, the broadest fitting tag first. Use only names from the list, and choose none if none fits.

- A tag fits only in its own meaning, as its description says: a tag for the programming language never fits a page about the snake or the island, even if they share a word.

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
/// stay on their own. O(n³) in the worst case, fine for a few thousand pages.
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
    loop {
        let mut best: Option<(usize, usize, f32)> = None;
        for a in (0..n).filter(|&a| members[a].is_some()) {
            for b in (a + 1..n).filter(|&b| members[b].is_some()) {
                if best.is_none_or(|(_, _, s)| sim[a][b] > s) {
                    best = Some((a, b, sim[a][b]));
                }
            }
        }
        let Some((a, b, s)) = best.filter(|(_, _, s)| *s >= threshold) else {
            break;
        };
        let _ = s;
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
    }
    let mut clusters: Vec<Vec<usize>> = members.into_iter().flatten().collect();
    clusters.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
    clusters
}

fn page_line(page: &AnalyzedPage) -> String {
    if page.tags.is_empty() {
        page.title.clone()
    } else {
        format!("{} (topics: {})", page.title, page.tags.join(", "))
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
    let descriptions: HashMap<String, String> = db
        .topic_tags()?
        .into_iter()
        .map(|t| (t.name, t.description.unwrap_or_default()))
        .collect();
    let mut topic_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for page in &pages {
        for tag in &page.tags {
            *topic_counts.entry(tag.as_str()).or_default() += 1;
        }
    }
    let mut vocabulary: BTreeMap<String, VocabTag> = BTreeMap::new();
    // Pages each vocabulary tag was made for, by index.
    let mut members: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, page) in pages.iter().enumerate() {
        for tag in &page.tags {
            if topic_counts
                .get(tag.as_str())
                .is_some_and(|n| *n >= settings.min_support)
            {
                members.entry(tag.clone()).or_default().push(i);
            }
        }
    }
    members.retain(|_, list| list.len() >= settings.min_support);
    for (name, n) in topic_counts
        .iter()
        .filter(|(name, _)| members.contains_key(**name))
    {
        let _ = n;
        vocabulary.insert(
            name.to_string(),
            VocabTag {
                name: name.to_string(),
                description: descriptions.get(*name).cloned().unwrap_or_default(),
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
    out.clusters = clusters.iter().map(Vec::len).collect();
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
        vocabulary.entry(name.clone()).or_insert(VocabTag {
            name,
            description: reply.description.trim().to_string(),
            source: Source::Cluster,
            pages: 0,
        });
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
        let Some(list) = members.get(&word) else { continue };
        // A word found to have one meaning is asked again only once it has
        // twice the pages.
        let key = format!("meanings:{word}");
        if let Some(n) = db
            .tag_decision(&key)?
            .and_then(|d| d.strip_prefix("same:").and_then(|n| n.parse::<usize>().ok()))
            && list.len() < n * 2
        {
            continue;
        }
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &i in list {
            groups.entry(cluster_of[i]).or_default().push(i);
        }
        if groups.len() < 2 {
            continue;
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
            word.clone(),
            shown
                .iter()
                .filter_map(|g| g.first().map(|p| p.title.clone()))
                .collect(),
        ));
        let things = match meanings::check(reviewer, &word, &shown).await {
            Ok(meanings::Verdict::Same) => {
                db.set_tag_decision(&key, &format!("same:{}", list.len()), "llm")?;
                continue;
            }
            Ok(meanings::Verdict::Several(things)) => things,
            Err(err) => {
                out.failed_requests.push(err);
                continue;
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
        let mut names = vec![word.clone()];
        let kept: Vec<usize> = things[main]
            .2
            .iter()
            .flat_map(|&g| groups[g].iter().copied())
            .collect();
        members.insert(word.clone(), kept);
        // The word now names that meaning: its description says so, so a page
        // about another meaning isn't given it.
        if let Some(tag) = vocabulary.get_mut(&word)
            && !things[main].1.is_empty()
        {
            tag.description = format!("{word}: {}", things[main].1);
        }
        for (t, (name, description, member_groups)) in things.into_iter().enumerate() {
            if t == main || name == word || names.contains(&name) {
                continue;
            }
            let list: Vec<usize> = member_groups
                .iter()
                .flat_map(|&g| groups[g].iter().copied())
                .collect();
            members.entry(name.clone()).or_default().extend(list);
            vocabulary.entry(name.clone()).or_insert(VocabTag {
                name: name.clone(),
                description,
                source: Source::Topic,
                pages: 0,
            });
            names.push(name);
        }
        out.ambiguous.push((word, names));
    }

    // Each page's tags, from the vocabulary tags nearest to it.
    let names: Vec<String> = vocabulary.keys().cloned().collect();
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
    let mut chosen: Vec<Vec<String>> = Vec::with_capacity(pages.len());
    for (page, vector) in pages.iter().zip(&page_vectors) {
        let mut ranked: Vec<(f32, usize)> = tag_vectors
            .iter()
            .enumerate()
            .map(|(i, t)| (similarity(vector, t), i))
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        let offered: Vec<&str> = ranked
            .iter()
            .take(settings.candidates)
            .map(|(_, i)| names[*i].as_str())
            .collect();
        let mut user = format!(
            "Page: {}\n{}\n\nTags to choose from:\n",
            page.title,
            truncate_chars(&page.summary, SUMMARY_CHARS)
        );
        for name in &offered {
            user.push_str(&format!("- {name}: {}\n", vocabulary[*name].description));
        }
        let picked: Vec<String> = match reviewer
            .ask_json(&system, &user, "page_tags", classify_schema.clone())
            .await
        {
            Ok(reply) => serde_json::from_value::<ClassifyReply>(reply)
                .map(|r| r.tags)
                .unwrap_or_default(),
            Err(err) => {
                out.failed_requests.push(format!("{err:#}"));
                Vec::new()
            }
        };
        let mut tags: Vec<String> = Vec::new();
        for raw in picked {
            if let Some(name) = normalize_name(&raw)
                && offered.contains(&name.as_str())
                && !tags.contains(&name)
                && tags.len() < settings.max_tags
            {
                tags.push(name);
            }
        }
        chosen.push(tags);
    }

    // Tags too few pages chose are dropped.
    let mut counts: HashMap<String, usize> = HashMap::new();
    for tags in &chosen {
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
        let tags: Vec<(String, String)> = tags
            .into_iter()
            .filter(|t| kept.contains(t))
            .map(|t| {
                let description = vocabulary[&t].description.clone();
                (t, description)
            })
            .collect();
        if tags.is_empty() {
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
            json!({"tags": ["coffee", "nowhere"]}),
            json!({"tags": ["python"]}),
            json!({"tags": ["python"]}),
            json!({"tags": ["python-snake"]}),
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
        assert_eq!(out.meaning_checks, 2);
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
        // The one-meaning verdict is remembered.
        assert_eq!(
            db.tag_decision("meanings:coffee").unwrap().as_deref(),
            Some("same:3")
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

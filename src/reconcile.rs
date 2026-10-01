//! The tag review: code finds candidate merges and splits, the model decides,
//! and the result is a list of changes for the user to approve.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::Result;
use serde::Deserialize;
use serde_json::json;

use crate::config::ReconcileConfig;
use crate::db::{Db, EmbeddingKind, SplitPart, TagChange, TagInfo};
use crate::embed::{BATCH_SIZE, Embedder, similarity};
use crate::extract::truncate_chars;
use crate::llm::Reviewer;
use crate::tags::{normalize_name, plural_variants};

/// Pages of one tag shown to the model to decide a split; the rest are
/// placed by embedding.
const SPLIT_SAMPLE: usize = 40;
/// Neighbours per tag checked for being the same tag.
const NEIGHBOURS: usize = 5;

/// A change for the user to approve, with how to show it and how to
/// remember that it was declined.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposed {
    pub change: TagChange,
    pub description: String,
    /// Recorded as declined when the user turns the change down.
    pub decision_key: String,
}

#[derive(Debug, Default)]
pub struct Review {
    pub proposed: Vec<Proposed>,
    pub merge_candidates: usize,
    pub split_candidates: usize,
}

/// Finds candidates, asks the model, and returns the changes it supports.
/// Without an embedder, only plural/singular pairs are checked and no splits
/// are proposed.
pub async fn review<R: Reviewer, E: Embedder>(
    db: &Db,
    reviewer: &R,
    embedder: Option<&E>,
    config: &ReconcileConfig,
    note: impl Fn(&str),
) -> Result<Review> {
    let tags: Vec<TagInfo> = db.topic_tags()?;
    let by_name: HashMap<&str, &TagInfo> = tags.iter().map(|t| (t.name.as_str(), t)).collect();
    // Names only: stored tag vectors include descriptions such as "first
    // used for: <page title>", which make unrelated tags from one page look alike.
    let mut name_vectors: HashMap<i64, Vec<f32>> = HashMap::new();
    if let Some(embedder) = embedder {
        for chunk in tags.chunks(BATCH_SIZE) {
            let texts: Vec<String> = chunk.iter().map(|t| t.name.replace('-', " ")).collect();
            for (tag, vector) in chunk.iter().zip(embedder.embed(&texts).await?) {
                name_vectors.insert(tag.id, vector);
            }
        }
    }

    let pairs = merge_candidates(db, &tags, &by_name, &name_vectors, config.similarity)?;
    let splits = match embedder {
        Some(embedder) => split_candidates(db, &tags, embedder.model(), config)?,
        None => Vec::new(),
    };
    let mut review = Review {
        merge_candidates: pairs.len(),
        split_candidates: splits.len(),
        ..Review::default()
    };
    if !pairs.is_empty() || !splits.is_empty() {
        note(&format!(
            "Reviewing tags: {} possible duplicates and {} possibly ambiguous tags.",
            pairs.len(),
            splits.len()
        ));
    }

    let mut merges = Vec::new();
    for batch in pairs.chunks(config.batch_size) {
        merges.extend(review_merges(db, reviewer, batch).await?);
    }
    let name_owner = |name: &str| db.tag_named(name).ok().flatten();
    review.proposed.extend(merge_changes(&tags, &merges, &name_owner));

    // A tag that a merge or rename changes isn't also split: the split's pages
    // and names were taken before the merge. A split part naming a tag that is
    // merged away or renamed follows it when the changes are applied, through
    // the alias the old name becomes, and only if that change is approved.
    let touched: HashSet<i64> = review
        .proposed
        .iter()
        .flat_map(|p| match p.change {
            TagChange::Merge { from, into } => vec![from, into],
            TagChange::Rename { tag, .. } => vec![tag],
            TagChange::Split { .. } => vec![],
        })
        .collect();
    for candidate in splits.iter().filter(|c| !touched.contains(&c.tag.id)) {
        if let Some(proposed) = review_split(db, reviewer, candidate, &by_name).await? {
            review.proposed.push(proposed);
        }
    }
    Ok(review)
}

/// "1 page", "3 pages".
fn pages(n: usize) -> String {
    if n == 1 {
        "1 page".into()
    } else {
        format!("{n} pages")
    }
}

fn pair_key(a: &str, b: &str) -> String {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    format!("merge:{a}|{b}")
}

/// Pairs of tags that may be the same: plural/singular spellings, and tags
/// whose embeddings are very similar. Locked tags and pairs already decided
/// are left out.
fn merge_candidates<'t>(
    db: &Db,
    tags: &'t [TagInfo],
    by_name: &HashMap<&str, &'t TagInfo>,
    vectors: &HashMap<i64, Vec<f32>>,
    threshold: f32,
) -> Result<Vec<(&'t TagInfo, &'t TagInfo)>> {
    let mut seen = HashSet::new();
    let mut pairs = Vec::new();
    let mut add = |a: &'t TagInfo, b: &'t TagInfo| -> Result<()> {
        if a.id == b.id || a.locked || b.locked {
            return Ok(());
        }
        let key = pair_key(&a.name, &b.name);
        if seen.insert(key.clone()) && db.tag_decision(&key)?.is_none() {
            pairs.push((a, b));
        }
        Ok(())
    };
    for tag in tags {
        for variant in plural_variants(&tag.name) {
            if let Some(other) = by_name.get(variant.as_str()) {
                add(tag, other)?;
            }
        }
    }
    for tag in tags {
        let Some(v) = vectors.get(&tag.id) else { continue };
        let mut nearest: Vec<(f32, &TagInfo)> = tags
            .iter()
            .filter(|t| t.id != tag.id)
            .filter_map(|t| Some((similarity(v, vectors.get(&t.id)?), t)))
            .filter(|(s, _)| *s >= threshold)
            .collect();
        nearest.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, other) in nearest.into_iter().take(NEIGHBOURS) {
            add(tag, other)?;
        }
    }
    Ok(pairs)
}

/// A tag that may cover two meanings: a sample of its pages for the model,
/// and the vectors of all its pages, to place the pages outside the sample.
struct SplitCandidate<'t> {
    tag: &'t TagInfo,
    sample: Vec<(i64, String, String)>,
    vectors: HashMap<i64, Vec<f32>>,
}

/// Tags on enough pages whose page vectors fall into two clearly separate groups.
fn split_candidates<'t>(
    db: &Db,
    tags: &'t [TagInfo],
    model: &str,
    config: &ReconcileConfig,
) -> Result<Vec<SplitCandidate<'t>>> {
    let page_vectors: HashMap<i64, Vec<f32>> =
        db.embeddings(EmbeddingKind::Page, model)?.into_iter().collect();
    let mut out = Vec::new();
    for tag in tags
        .iter()
        .filter(|t| !t.locked && t.pages >= config.split_min_pages)
    {
        if db.tag_decision(&format!("split:{}", tag.name))?.is_some() {
            continue;
        }
        let pages = db.tag_pages(tag.id)?;
        let vectors: Vec<(i64, &Vec<f32>)> = pages
            .iter()
            .filter_map(|(id, ..)| Some((*id, page_vectors.get(id)?)))
            .collect();
        if vectors.len() < config.split_min_pages {
            continue;
        }
        let groups = two_groups(&vectors.iter().map(|(_, v)| v.as_slice()).collect::<Vec<_>>());
        let Some((labels, centroid_similarity)) = groups else {
            continue;
        };
        if centroid_similarity > config.split_similarity {
            continue;
        }
        // The sample takes pages from both groups, so the model sees both meanings.
        let mut sample_ids = Vec::new();
        for group in [0, 1] {
            sample_ids.extend(
                vectors
                    .iter()
                    .zip(&labels)
                    .filter(|(_, l)| **l == group)
                    .map(|((id, _), _)| *id)
                    .take(SPLIT_SAMPLE / 2),
            );
        }
        let sample = pages
            .iter()
            .filter(|(id, ..)| sample_ids.contains(id))
            .cloned()
            .collect();
        let vectors = vectors.into_iter().map(|(id, v)| (id, v.clone())).collect();
        out.push(SplitCandidate { tag, sample, vectors });
    }
    Ok(out)
}

/// Splits unit vectors into two groups (2-means, cosine). Returns each
/// vector's group and how similar the two group centres are. A group may be a
/// single page: a tag on one page it doesn't describe is worth a look too.
fn two_groups(vectors: &[&[f32]]) -> Option<(Vec<usize>, f32)> {
    if vectors.len() < 3 {
        return None;
    }
    // Start from the vector least like the first, so the seeds are far apart.
    let far = (1..vectors.len())
        .min_by(|&a, &b| similarity(vectors[0], vectors[a]).total_cmp(&similarity(vectors[0], vectors[b])))?;
    let mut centres = [vectors[0].to_vec(), vectors[far].to_vec()];
    let mut labels = vec![0; vectors.len()];
    for _ in 0..10 {
        for (label, v) in labels.iter_mut().zip(vectors) {
            *label = usize::from(similarity(v, &centres[1]) > similarity(v, &centres[0]));
        }
        for (group, centre) in centres.iter_mut().enumerate() {
            let members: Vec<&&[f32]> = vectors
                .iter()
                .zip(&labels)
                .filter(|(_, l)| **l == group)
                .map(|(v, _)| v)
                .collect();
            if members.is_empty() {
                return None;
            }
            *centre = normalized_mean(&members.iter().map(|v| **v).collect::<Vec<_>>());
        }
    }
    let sizes = [
        labels.iter().filter(|l| **l == 0).count(),
        labels.iter().filter(|l| **l == 1).count(),
    ];
    if sizes.contains(&0) {
        return None;
    }
    Some((labels, similarity(&centres[0], &centres[1])))
}

fn normalized_mean(vectors: &[&[f32]]) -> Vec<f32> {
    let mut mean = vec![0.0f32; vectors[0].len()];
    for v in vectors {
        for (m, x) in mean.iter_mut().zip(v.iter()) {
            *m += x;
        }
    }
    let norm = mean.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    mean.iter_mut().for_each(|x| *x /= norm);
    mean
}

/// A tag as the model sees it: name, page count, description, example pages.
fn describe_tag(db: &Db, tag: &TagInfo) -> Result<String> {
    let examples: Vec<String> = db
        .tag_pages(tag.id)?
        .into_iter()
        .take(2)
        .map(|(_, title, _)| format!("\"{title}\""))
        .collect();
    let mut text = format!("{} ({}", tag.name, pages(tag.pages));
    if !examples.is_empty() {
        text.push_str(&format!(", e.g. {}", examples.join(", ")));
    }
    text.push(')');
    if let Some(d) = tag.description.as_deref().filter(|d| !d.is_empty()) {
        text.push_str(&format!(": {d}"));
    }
    Ok(text)
}

const MERGE_SYSTEM: &str = "You maintain the tags of a personal archive of web pages. For each numbered pair of tags, decide whether the two tags are the same tag under two names.

- merge is true only if the two names are interchangeable: every page that deserves one deserves the other. That covers synonyms, spelling variants, singular and plural, and abbreviations (ml and machine-learning, board-game and board-games).
- merge is false for a broader and a narrower topic (marathon and olympic-marathon, music and folk-music), for two subtopics of one field (coffee-brewing and espresso-brewing), for a word and its other meaning (go and go-game), and for topics that are merely related or appear on the same pages (tea and history).
- Most pairs here are not the same tag. When in doubt, merge is false.
- When merging, keep is the name for the merged tag: usually one of the two, or a clearer lowercase kebab-case name for both.

Reply with a JSON object: {\"decisions\": [{\"pair\": 1, \"merge\": true, \"keep\": \"name\"}, ...]}, one decision per pair.";

#[derive(Deserialize)]
struct MergeReply {
    decisions: Vec<MergeDecision>,
}

#[derive(Deserialize)]
struct MergeDecision {
    pair: usize,
    merge: bool,
    #[serde(default)]
    keep: String,
}

/// (tag a, tag b, name to keep) for each pair the model wants merged. Pairs
/// it keeps apart are remembered, so they aren't asked about again.
async fn review_merges<R: Reviewer>(
    db: &Db,
    reviewer: &R,
    batch: &[(&TagInfo, &TagInfo)],
) -> Result<Vec<(i64, i64, String)>> {
    let mut user = String::from("Pairs of tags:\n");
    for (i, (a, b)) in batch.iter().enumerate() {
        user.push_str(&format!(
            "{}. {}\n   {}\n",
            i + 1,
            describe_tag(db, a)?,
            describe_tag(db, b)?
        ));
    }
    let schema = json!({
        "type": "object", "additionalProperties": false, "required": ["decisions"],
        "properties": {"decisions": {"type": "array", "items": {
            "type": "object", "additionalProperties": false, "required": ["pair", "merge", "keep"],
            "properties": {"pair": {"type": "integer"}, "merge": {"type": "boolean"}, "keep": {"type": "string"}}
        }}}
    });
    let reply: MergeReply = serde_json::from_value(
        reviewer
            .ask_json(MERGE_SYSTEM, &user, "tag_merges", schema)
            .await?,
    )?;
    let mut merges = Vec::new();
    for decision in reply.decisions {
        let Some((a, b)) = batch.get(decision.pair.wrapping_sub(1)) else {
            continue;
        };
        if decision.merge {
            let keep = normalize_name(&decision.keep).unwrap_or_else(|| a.name.clone());
            merges.push((a.id, b.id, keep));
        } else {
            db.set_tag_decision(&pair_key(&a.name, &b.name), "separate", "llm")?;
        }
    }
    Ok(merges)
}

/// Turns the model's merges into changes: tags merged in chains end up in one
/// tag, named by the most voted `keep`. A name that isn't one of the group's
/// tags renames the group's most used tag.
/// `name_owner` finds the tag that already uses a name, as a tag or an alias,
/// including tags on no finished page.
fn merge_changes(
    tags: &[TagInfo],
    merges: &[(i64, i64, String)],
    name_owner: &dyn Fn(&str) -> Option<i64>,
) -> Vec<Proposed> {
    let by_id: HashMap<i64, &TagInfo> = tags.iter().map(|t| (t.id, t)).collect();
    let by_name: HashMap<&str, &TagInfo> = tags.iter().map(|t| (t.name.as_str(), t)).collect();
    // Union-find over tag ids.
    let mut parent: HashMap<i64, i64> = HashMap::new();
    fn root(parent: &mut HashMap<i64, i64>, id: i64) -> i64 {
        let p = *parent.get(&id).unwrap_or(&id);
        if p == id {
            return id;
        }
        let r = root(parent, p);
        parent.insert(id, r);
        r
    }
    let mut votes: HashMap<i64, Vec<String>> = HashMap::new();
    for (a, b, keep) in merges {
        // A kept name that is another existing tag joins that tag to the
        // group, unless it is locked.
        let joined = by_name.get(keep.as_str()).filter(|t| !t.locked).map(|t| t.id);
        for other in [Some(*b), joined].into_iter().flatten() {
            let (ra, rb) = (root(&mut parent, *a), root(&mut parent, other));
            if ra != rb {
                parent.insert(rb, ra);
                let moved = votes.remove(&rb).unwrap_or_default();
                votes.entry(ra).or_default().extend(moved);
            }
        }
        let r = root(&mut parent, *a);
        votes.entry(r).or_default().push(keep.clone());
    }
    let mut groups: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    let ids: Vec<i64> = parent.keys().copied().collect();
    for id in ids {
        let r = root(&mut parent, id);
        groups.entry(r).or_default().push(id);
    }

    let mut out = Vec::new();
    // New names already given to another group in this review.
    let mut renamed_to: HashSet<String> = HashSet::new();
    for (r, mut members) in groups {
        members.sort();
        members.dedup();
        if !members.contains(&r) {
            members.push(r);
        }
        let members: Vec<&TagInfo> = members.iter().filter_map(|id| by_id.get(id).copied()).collect();
        if members.len() < 2 {
            continue;
        }
        // The most voted name; ties go to an existing tag with more pages.
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for name in votes.get(&r).into_iter().flatten() {
            *counts.entry(name.as_str()).or_default() += 1;
        }
        let pages_of = |name: &str| members.iter().find(|t| t.name == name).map_or(0, |t| t.pages);
        let biggest_name = || {
            members
                .iter()
                .max_by_key(|t| t.pages)
                .expect("two members")
                .name
                .clone()
        };
        let mut keep = counts
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then(pages_of(a.0).cmp(&pages_of(b.0))))
            .map(|(name, _)| name.to_string())
            .unwrap_or_else(biggest_name);
        // A new name already used by a tag outside the group, or given to
        // another group, would clash.
        if !members.iter().any(|t| t.name == keep)
            && (renamed_to.contains(&keep)
                || name_owner(&keep).is_some_and(|id| !members.iter().any(|t| t.id == id)))
        {
            keep = biggest_name();
        }
        let into = match members.iter().find(|t| t.name == keep) {
            Some(existing) => *existing,
            None => {
                let biggest = *members.iter().max_by_key(|t| t.pages).expect("two members");
                out.push(Proposed {
                    change: TagChange::Rename {
                        tag: biggest.id,
                        name: keep.clone(),
                    },
                    description: format!("rename {} → {keep} ({})", biggest.name, pages(biggest.pages)),
                    decision_key: format!("rename:{}>{keep}", biggest.name),
                });
                renamed_to.insert(keep.clone());
                biggest
            }
        };
        for from in members.iter().filter(|t| t.id != into.id) {
            out.push(Proposed {
                change: TagChange::Merge {
                    from: from.id,
                    into: into.id,
                },
                description: format!(
                    "merge  {} → {keep} ({} + {} pages)",
                    from.name, from.pages, into.pages
                ),
                decision_key: pair_key(&from.name, &into.name),
            });
        }
    }
    out
}

const SPLIT_SYSTEM: &str = "You maintain the tags of a personal archive of web pages. Sometimes one tag name ends up with two unrelated meanings, like rust for the programming language and for iron corrosion. Given the pages that have a tag, decide whether that happened.

- First, in meanings, list the distinct meanings the tag name has across these pages. A meaning is what the word itself refers to, not the subject of a page: history pages about marathons, Kyoto and the French Revolution all use history in one meaning; open-source pages about Python and Go use open-source in one meaning. Different subtopics, eras, places or projects are one meaning.
- Set split to true only if there are two or more unrelated meanings, so that a person looking for one of them would not want the others. Otherwise set split to false and leave tags and assignments empty. Most tags have one meaning.
- When splitting, list one tag per meaning, not one per page, each with a one-line description, and assign every listed page to exactly one of them. Keep the tag's own name for its most common meaning. For each other meaning, use a tag these pages already have, or a new lowercase kebab-case name qualified so it is unambiguous on its own (rust-corrosion).

Reply with a JSON object: {\"meanings\": [\"...\"], \"split\": true, \"tags\": [{\"name\": \"...\", \"description\": \"...\"}], \"assignments\": [{\"page\": 12, \"tag\": \"...\"}]}.";

#[derive(Deserialize)]
struct SplitReply {
    split: bool,
    #[serde(default)]
    tags: Vec<SplitTag>,
    #[serde(default)]
    assignments: Vec<Assignment>,
}

#[derive(Deserialize)]
struct SplitTag {
    name: String,
    #[serde(default)]
    description: String,
}

#[derive(Deserialize)]
struct Assignment {
    page: i64,
    tag: String,
}

async fn review_split<R: Reviewer>(
    db: &Db,
    reviewer: &R,
    candidate: &SplitCandidate<'_>,
    by_name: &HashMap<&str, &TagInfo>,
) -> Result<Option<Proposed>> {
    let SplitCandidate { tag, sample, vectors } = candidate;
    let mut user = format!("Tag: {}\nPages with this tag:\n", describe_tag(db, tag)?);
    for (id, title, summary) in sample {
        user.push_str(&format!(
            "- page {id}: {title}. {}\n",
            truncate_chars(summary, 200)
        ));
    }
    let schema = json!({
        "type": "object", "additionalProperties": false, "required": ["meanings", "split", "tags", "assignments"],
        "properties": {
            "meanings": {"type": "array", "items": {"type": "string"}},
            "split": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "object", "additionalProperties": false,
                "required": ["name", "description"],
                "properties": {"name": {"type": "string"}, "description": {"type": "string"}}}},
            "assignments": {"type": "array", "items": {"type": "object", "additionalProperties": false,
                "required": ["page", "tag"],
                "properties": {"page": {"type": "integer"}, "tag": {"type": "string"}}}}
        }
    });
    let reply: SplitReply = serde_json::from_value(
        reviewer
            .ask_json(SPLIT_SYSTEM, &user, "tag_split", schema)
            .await?,
    )?;
    let key = format!("split:{}", tag.name);
    if !reply.split {
        db.set_tag_decision(&key, "single-meaning", "llm")?;
        return Ok(None);
    }
    // An unusable split isn't remembered, so the next review asks again.
    let Some(mut parts) = split_parts(sample, &reply, by_name) else {
        return Ok(None);
    };
    place_remaining_pages(&mut parts, vectors);
    let description = format!(
        "split  {} → {}",
        tag.name,
        parts
            .iter()
            .map(|p| format!("{} ({})", p.name, pages(p.pages.len())))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(Some(Proposed {
        change: TagChange::Split {
            tag: tag.id,
            into: parts,
        },
        description,
        decision_key: key,
    }))
}

/// The split's parts with the sampled pages, if the reply is usable: two or
/// more valid names, none the tag's own, and every sampled page assigned.
fn split_parts(
    sample: &[(i64, String, String)],
    reply: &SplitReply,
    by_name: &HashMap<&str, &TagInfo>,
) -> Option<Vec<SplitPart>> {
    let mut parts: Vec<SplitPart> = Vec::new();
    for t in &reply.tags {
        let name = normalize_name(&t.name)?;
        if parts.iter().any(|p| p.name == name) {
            return None;
        }
        // An existing tag keeps its own description.
        let description = if by_name.contains_key(name.as_str()) {
            String::new()
        } else {
            t.description.trim().to_string()
        };
        parts.push(SplitPart {
            name,
            description,
            pages: Vec::new(),
        });
    }
    if parts.len() < 2 {
        return None;
    }
    for (page, ..) in sample {
        let assigned = reply.assignments.iter().find(|a| a.page == *page)?;
        let name = normalize_name(&assigned.tag)?;
        parts.iter_mut().find(|p| p.name == name)?.pages.push(*page);
    }
    parts.retain(|p| !p.pages.is_empty());
    (parts.len() >= 2).then_some(parts)
}

/// Pages that weren't in the sample go to the part whose pages they're most
/// like.
fn place_remaining_pages(parts: &mut [SplitPart], vectors: &HashMap<i64, Vec<f32>>) {
    let placed: HashSet<i64> = parts.iter().flat_map(|p| p.pages.iter().copied()).collect();
    let centres: Vec<Option<Vec<f32>>> = parts
        .iter()
        .map(|p| {
            let members: Vec<&[f32]> = p
                .pages
                .iter()
                .filter_map(|id| vectors.get(id).map(Vec::as_slice))
                .collect();
            (!members.is_empty()).then(|| normalized_mean(&members))
        })
        .collect();
    let mut rest: Vec<(&i64, &Vec<f32>)> = vectors.iter().filter(|(id, _)| !placed.contains(id)).collect();
    rest.sort_by_key(|(id, _)| **id);
    for (id, v) in rest {
        let best = (0..parts.len())
            .filter_map(|i| Some((i, similarity(v, centres[i].as_ref()?))))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((best, _)) = best {
            parts[best].pages.push(*id);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use serde_json::Value;

    use super::*;
    use crate::db::PageResult;

    /// Replies in order and records the prompts it was given.
    struct FakeReviewer {
        replies: RefCell<Vec<Value>>,
        prompts: RefCell<Vec<String>>,
    }

    impl FakeReviewer {
        fn new(replies: Vec<Value>) -> Self {
            Self {
                replies: RefCell::new(replies),
                prompts: RefCell::default(),
            }
        }
    }

    impl Reviewer for FakeReviewer {
        async fn ask_json(&self, _system: &str, user: &str, _name: &str, _schema: Value) -> Result<Value> {
            self.prompts.borrow_mut().push(user.to_string());
            Ok(self.replies.borrow_mut().remove(0))
        }
    }

    /// Embeds tag names from a fixed table; names not in it get a vector of
    /// their own, unlike any other.
    struct NameEmbedder(Vec<(&'static str, [f32; 3])>);

    impl Embedder for NameEmbedder {
        fn model(&self) -> &str {
            "m"
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .enumerate()
                .map(|(i, text)| match self.0.iter().find(|(name, _)| *name == text) {
                    Some((_, v)) => {
                        let mut out = unit(v);
                        out.resize(3 + texts.len(), 0.0);
                        out
                    }
                    None => {
                        let mut out = vec![0.0; 3 + texts.len()];
                        out[3 + i] = 1.0;
                        out
                    }
                })
                .collect())
        }
    }

    fn no_names() -> NameEmbedder {
        NameEmbedder(Vec::new())
    }

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    /// One page per entry: (tags, page vector).
    fn archive(pages: &[(&[&str], [f32; 3])]) -> Db {
        let mut db = Db::open_in_memory().unwrap();
        for (i, (tags, vector)) in pages.iter().enumerate() {
            let url = format!("https://p{i}.com/");
            db.add_page(&url, &url, None, "import").unwrap();
            let id = db
                .pending_pages()
                .unwrap()
                .iter()
                .find(|p| p.url == url)
                .unwrap()
                .id;
            let resolved: Vec<(String, i64)> = tags
                .iter()
                .map(|name| {
                    let tag = db
                        .find_tag(name)
                        .unwrap()
                        .map(|(id, _)| id)
                        .unwrap_or_else(|| db.create_tag(name, None).unwrap());
                    (name.to_string(), tag)
                })
                .collect();
            let title = format!("Page {i}");
            db.save_result(
                id,
                &PageResult {
                    title: &title,
                    summary: "S.",
                    lang: None,
                    tags: &resolved,
                    page_title: None,
                },
            )
            .unwrap();
            db.put_embedding(EmbeddingKind::Page, id, "m", &unit(vector))
                .unwrap();
        }
        db
    }

    fn tag_id(db: &Db, name: &str) -> i64 {
        db.find_tag(name).unwrap().unwrap().0
    }

    fn info(id: i64, name: &str, pages: usize) -> TagInfo {
        TagInfo {
            id,
            name: name.into(),
            description: None,
            locked: false,
            pages,
        }
    }

    #[test]
    fn chained_merges_end_in_one_tag() {
        let tags = [
            info(1, "ml", 2),
            info(2, "machine-learning", 5),
            info(3, "machinelearning-basics", 1),
        ];
        let merges = [
            (1, 2, "machine-learning".to_string()),
            (3, 1, "machine-learning".to_string()),
        ];
        let changes: Vec<TagChange> = merge_changes(&tags, &merges, &|_| None)
            .into_iter()
            .map(|p| p.change)
            .collect();
        assert_eq!(changes.len(), 2);
        assert!(changes.contains(&TagChange::Merge { from: 1, into: 2 }));
        assert!(changes.contains(&TagChange::Merge { from: 3, into: 2 }));
    }

    #[test]
    fn a_better_name_renames_the_most_used_tag() {
        let tags = [info(1, "js", 4), info(2, "java-script", 1)];
        let proposed = merge_changes(&tags, &[(1, 2, "javascript".to_string())], &|_| None);
        let changes: Vec<TagChange> = proposed.iter().map(|p| p.change.clone()).collect();
        assert_eq!(
            changes,
            [
                TagChange::Rename {
                    tag: 1,
                    name: "javascript".into()
                },
                TagChange::Merge { from: 2, into: 1 }
            ]
        );
        assert_eq!(proposed[0].description, "rename js → javascript (4 pages)");
    }

    #[test]
    fn two_groups_dont_get_the_same_new_name() {
        let tags = [
            info(1, "js", 4),
            info(2, "java-script", 1),
            info(3, "ecmascript", 2),
            info(4, "es", 1),
        ];
        let merges = [(1, 2, "javascript".to_string()), (3, 4, "javascript".to_string())];
        let renames: Vec<String> = merge_changes(&tags, &merges, &|_| None)
            .into_iter()
            .filter_map(|p| match p.change {
                TagChange::Rename { name, .. } => Some(name),
                _ => None,
            })
            .collect();
        assert_eq!(renames, ["javascript"]);
    }

    #[test]
    fn a_locked_tag_isnt_pulled_into_a_merge() {
        let mut locked = info(3, "javascript", 1);
        locked.locked = true;
        let tags = [info(1, "js", 4), info(2, "java-script", 1), locked];
        let owner = |name: &str| (name == "javascript").then_some(3);
        let changes: Vec<TagChange> = merge_changes(&tags, &[(1, 2, "javascript".to_string())], &owner)
            .into_iter()
            .map(|p| p.change)
            .collect();
        assert_eq!(changes, [TagChange::Merge { from: 2, into: 1 }]);
    }

    #[test]
    fn a_name_used_elsewhere_is_not_taken() {
        let tags = [info(1, "js", 4), info(2, "java-script", 1)];
        // An unused tag, or another tag's alias, is already called javascript.
        let owner = |name: &str| (name == "javascript").then_some(9);
        let changes: Vec<TagChange> = merge_changes(&tags, &[(1, 2, "javascript".to_string())], &owner)
            .into_iter()
            .map(|p| p.change)
            .collect();
        assert_eq!(changes, [TagChange::Merge { from: 2, into: 1 }]);
    }

    #[tokio::test]
    async fn proposes_merges_the_model_confirms_and_remembers_the_rest() {
        let db = archive(&[
            (&["board-games", "ml"], [1.0, 0.0, 0.0]),
            (&["board-game", "machine-learning"], [1.0, 0.1, 0.0]),
            (&["react", "react-native"], [0.0, 1.0, 0.0]),
        ]);
        // ml and machine-learning, and react and react-native, have close names in embedding space.
        let names = NameEmbedder(vec![
            ("ml", [0.0, 0.0, 1.0]),
            ("machine learning", [0.05, 0.0, 1.0]),
            ("react", [1.0, 1.0, 0.0]),
            ("react native", [1.0, 0.95, 0.0]),
        ]);
        let reviewer = FakeReviewer::new(vec![json!({"decisions": [
            {"pair": 1, "merge": true, "keep": "board-games"},
            {"pair": 2, "merge": true, "keep": "machine-learning"},
            {"pair": 3, "merge": false, "keep": ""}
        ]})]);
        let config = ReconcileConfig {
            split_min_pages: 99,
            ..ReconcileConfig::default()
        };
        let review = review(&db, &reviewer, Some(&names), &config, |_| {})
            .await
            .unwrap();
        assert_eq!(review.merge_candidates, 3);
        let descriptions: Vec<&str> = review.proposed.iter().map(|p| p.description.as_str()).collect();
        assert_eq!(
            descriptions,
            [
                "merge  board-game → board-games (1 + 1 pages)",
                "merge  ml → machine-learning (1 + 1 pages)"
            ]
        );
        let prompt = reviewer.prompts.borrow()[0].clone();
        assert!(
            prompt.contains(
                "1. board-game (1 page, e.g. \"Page 1\")\n   board-games (1 page, e.g. \"Page 0\")"
            ),
            "{prompt}"
        );

        // react / react-native was judged different: it isn't asked about again.
        let again = FakeReviewer::new(vec![json!({"decisions": [
            {"pair": 1, "merge": true, "keep": "board-games"},
            {"pair": 2, "merge": true, "keep": "machine-learning"}
        ]})]);
        let review = super::review(&db, &again, Some(&names), &config, |_| {})
            .await
            .unwrap();
        assert_eq!(review.merge_candidates, 2);
    }

    #[tokio::test]
    async fn splits_a_tag_whose_pages_fall_apart() {
        let db = archive(&[
            (&["rust"], [1.0, 0.0, 0.0]),
            (&["rust"], [1.0, 0.1, 0.0]),
            (&["rust"], [0.0, 0.0, 1.0]),
            (&["rust"], [0.1, 0.0, 1.0]),
            (&["python"], [1.0, 0.0, 0.0]),
            (&["python"], [1.0, 0.05, 0.0]),
            (&["python"], [0.95, 0.0, 0.0]),
            (&["python"], [1.0, 0.0, 0.05]),
        ]);
        let pages: Vec<i64> = db
            .tag_pages(tag_id(&db, "rust"))
            .unwrap()
            .into_iter()
            .map(|(id, ..)| id)
            .collect();
        let reviewer = FakeReviewer::new(vec![json!({
            "split": true,
            "tags": [{"name": "Rust Programming", "description": "The language"}, {"name": "rust-corrosion", "description": "Iron oxide"}],
            "assignments": [
                {"page": pages[0], "tag": "rust-programming"}, {"page": pages[1], "tag": "rust-programming"},
                {"page": pages[2], "tag": "rust-corrosion"}, {"page": pages[3], "tag": "rust-corrosion"}
            ]
        })]);
        let review = review(
            &db,
            &reviewer,
            Some(&no_names()),
            &ReconcileConfig::default(),
            |_| {},
        )
        .await
        .unwrap();
        // python's pages are all alike: only rust is a candidate.
        assert_eq!(review.split_candidates, 1);
        assert_eq!(review.proposed.len(), 1);
        assert_eq!(
            review.proposed[0].description,
            "split  rust → rust-programming (2 pages), rust-corrosion (2 pages)"
        );
    }

    #[tokio::test]
    async fn an_unusable_split_reply_proposes_nothing() {
        let db = archive(&[
            (&["rust"], [1.0, 0.0, 0.0]),
            (&["rust"], [1.0, 0.1, 0.0]),
            (&["rust"], [0.0, 0.0, 1.0]),
            (&["rust"], [0.1, 0.0, 1.0]),
        ]);
        // Only one page assigned.
        let pages: Vec<i64> = db
            .tag_pages(tag_id(&db, "rust"))
            .unwrap()
            .into_iter()
            .map(|(id, ..)| id)
            .collect();
        let reviewer = FakeReviewer::new(vec![json!({
            "split": true,
            "tags": [{"name": "rust-programming", "description": ""}, {"name": "rust-corrosion", "description": ""}],
            "assignments": [{"page": pages[0], "tag": "rust-programming"}]
        })]);
        let review = review(
            &db,
            &reviewer,
            Some(&no_names()),
            &ReconcileConfig::default(),
            |_| {},
        )
        .await
        .unwrap();
        assert!(review.proposed.is_empty());
        // Not remembered: the model did say the tag should be split.
        assert_eq!(db.tag_decision("split:rust").unwrap(), None);
    }

    #[tokio::test]
    async fn a_merged_tag_is_not_also_split() {
        let db = archive(&[
            (&["rust"], [1.0, 0.0, 0.0]),
            (&["rust"], [1.0, 0.1, 0.0]),
            (&["rust"], [0.0, 0.0, 1.0]),
            (&["rusts"], [0.1, 0.0, 1.0]),
        ]);
        // Only the merge is asked; a split request would find no reply.
        let reviewer = FakeReviewer::new(vec![json!({"decisions": [
            {"pair": 1, "merge": true, "keep": "rust"}
        ]})]);
        let review = review(
            &db,
            &reviewer,
            Some(&no_names()),
            &ReconcileConfig {
                split_min_pages: 3,
                ..ReconcileConfig::default()
            },
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(review.split_candidates, 1);
        let changes: Vec<TagChange> = review.proposed.into_iter().map(|p| p.change).collect();
        assert_eq!(
            changes,
            [TagChange::Merge {
                from: tag_id(&db, "rusts"),
                into: tag_id(&db, "rust")
            }]
        );
    }

    #[test]
    fn two_groups_and_placing_the_rest() {
        let vectors: Vec<Vec<f32>> = [[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]
            .iter()
            .map(|v| unit(v))
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let (labels, centres) = two_groups(&refs).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
        assert!(centres < 0.2, "{centres}");

        let mut parts = vec![
            SplitPart {
                name: "a".into(),
                description: String::new(),
                pages: vec![1],
            },
            SplitPart {
                name: "b".into(),
                description: String::new(),
                pages: vec![3],
            },
        ];
        let all: HashMap<i64, Vec<f32>> = (1..=4).zip(vectors).collect();
        place_remaining_pages(&mut parts, &all);
        assert_eq!(
            (parts[0].pages.clone(), parts[1].pages.clone()),
            (vec![1, 2], vec![3, 4])
        );
    }
}

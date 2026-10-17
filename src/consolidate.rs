//! Consolidating the tag vocabulary: near-duplicates and tags too narrow to
//! stand on their own are folded into broader tags. Code offers each tag the
//! tags whose pages are about the same things; the model decides. It runs
//! after merges and splits and before the tag tree, so the tree is built
//! from the consolidated tags.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::Result;
use serde::Deserialize;
use serde_json::json;

use crate::db::{Db, EmbeddingKind, TagChange, TagInfo};
use crate::embed::{Embedder, similarity};
use crate::llm::Reviewer;
use crate::reconcile::{Proposed, count, describe_tag, normalized_mean, pages};
use crate::tags::normalize_name;

/// Tags decided per request.
const BATCH: usize = 10;
/// Broader tags offered for one tag.
const CANDIDATES: usize = 3;
/// Tag names listed in a proposal's description; more are counted.
const NAMES_SHOWN: usize = 8;

const SYSTEM: &str = "You keep the tag vocabulary of a personal archive of web pages small and useful, so that each tag gathers several pages. Each numbered tag comes with the tags it could be folded into: tags whose pages are about the same things. Decide for each whether it stays a tag of its own or is folded into one of them.

- Fold it (into: that tag's name) when that tag means the same, or covers it and the tag is a detail too narrow to look for on its own: python-releases into python, espresso-brewing into coffee, transformer-architecture into deep-learning.
- Keep it (into: empty) when it names a distinct topic someone would browse by, such as a language, a place, a species, a field, a product or a hobby, even if few pages have it yet.
- Keep it when the other tag is a different meaning of the same word: never fold python-snake into python-programming.
- When in doubt, keep the tag.

Reply with a JSON object: {\"decisions\": [{\"tag\": 1, \"into\": \"\"}, {\"tag\": 2, \"into\": \"one-of-its-candidates\"}, ...]}, one decision per tag.";

#[derive(Deserialize)]
struct Reply {
    decisions: Vec<Decision>,
}

#[derive(Deserialize)]
struct Decision {
    tag: usize,
    #[serde(default)]
    into: String,
}

/// What the consolidation step proposes.
#[derive(Debug, Default)]
pub struct Consolidation {
    pub proposed: Vec<Proposed>,
    /// Tags the model was asked about.
    pub checked: usize,
    pub failed_requests: Vec<String>,
}

fn keep_key(name: &str) -> String {
    format!("consolidate:{name}")
}

fn fold_key(from: &str, into: &str) -> String {
    format!("fold:{from}>{into}")
}

/// A tag kept before is asked again only when it has a candidate it wasn't
/// offered then.
fn kept_with(decision: &str) -> HashSet<&str> {
    decision
        .strip_prefix("keep:")
        .map(|names| names.split(',').filter(|n| !n.is_empty()).collect())
        .unwrap_or_default()
}

/// For each tag, the tags it could be folded into: at least as many pages
/// (a big tag is never folded into a small one), and pages at least
/// `threshold` alike, most alike first.
fn candidates<'t>(
    tags: &'t [TagInfo],
    centroids: &HashMap<i64, Vec<f32>>,
    threshold: f32,
) -> HashMap<i64, Vec<&'t TagInfo>> {
    let mut out = HashMap::new();
    for tag in tags {
        let Some(own) = centroids.get(&tag.id) else {
            continue;
        };
        let mut near: Vec<(f32, &TagInfo)> = tags
            .iter()
            .filter(|other| other.id != tag.id && other.pages >= tag.pages)
            .filter_map(|other| Some((similarity(own, centroids.get(&other.id)?), other)))
            .filter(|(s, _)| *s >= threshold)
            .collect();
        near.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.pages.cmp(&a.1.pages)));
        near.truncate(CANDIDATES);
        if !near.is_empty() {
            out.insert(tag.id, near.into_iter().map(|(_, t)| t).collect());
        }
    }
    out
}

/// Asks about every topic tag with a possible broader tag, and proposes one
/// fold per broader tag.
pub async fn consolidate<R: Reviewer, E: Embedder>(
    db: &Db,
    reviewer: &R,
    embedder: &E,
    threshold: f32,
    note: impl Fn(&str),
) -> Result<Consolidation> {
    let mut out = Consolidation::default();
    let tags = db.topic_tags()?;
    let page_vectors: HashMap<i64, Vec<f32>> = db
        .embeddings(EmbeddingKind::Page, embedder.model())?
        .into_iter()
        .collect();
    let mut centroids = HashMap::new();
    for tag in &tags {
        let vectors: Vec<&[f32]> = db
            .tag_pages(tag.id)?
            .iter()
            .filter_map(|(id, ..)| page_vectors.get(id).map(Vec::as_slice))
            .collect();
        if !vectors.is_empty() {
            centroids.insert(tag.id, normalized_mean(&vectors));
        }
    }
    let offered = candidates(&tags, &centroids, threshold);

    let mut asked: Vec<(&TagInfo, &Vec<&TagInfo>)> = Vec::new();
    for tag in tags.iter().filter(|t| !t.locked) {
        let Some(into) = offered.get(&tag.id) else {
            continue;
        };
        if let Some(decision) = db.tag_decision(&keep_key(&tag.name))? {
            let before = kept_with(&decision);
            if into.iter().all(|t| before.contains(t.name.as_str())) {
                continue;
            }
        }
        asked.push((tag, into));
    }
    if asked.is_empty() {
        return Ok(out);
    }
    out.checked = asked.len();
    note(&format!(
        "Consolidating the tags: {} could be folded into broader ones.",
        count(asked.len(), "tag", "tags")
    ));

    let schema = json!({
        "type": "object", "additionalProperties": false, "required": ["decisions"],
        "properties": {"decisions": {"type": "array", "items": {
            "type": "object", "additionalProperties": false, "required": ["tag", "into"],
            "properties": {"tag": {"type": "integer"}, "into": {"type": "string"}}
        }}}
    });
    // tag name -> the tag it folds into.
    let mut folds: BTreeMap<String, String> = BTreeMap::new();
    for batch in asked.chunks(BATCH) {
        let mut user = String::from("Tags to decide, each with the tags it could be folded into:\n");
        for (i, (tag, into)) in batch.iter().enumerate() {
            user.push_str(&format!("{}. {}\n", i + 1, describe_tag(db, tag)?));
            for other in *into {
                user.push_str(&format!("   - could fold into {}\n", describe_tag(db, other)?));
            }
        }
        let reply = match reviewer
            .ask_json(SYSTEM, &user, "tag_consolidation", schema.clone())
            .await
        {
            Ok(reply) => reply,
            Err(err) => {
                out.failed_requests.push(format!("{err:#}"));
                continue;
            }
        };
        let Ok(reply) = serde_json::from_value::<Reply>(reply) else {
            out.failed_requests
                .push("the model's consolidation reply was unusable".into());
            continue;
        };
        let mut answered = HashSet::new();
        for decision in reply.decisions {
            let Some((tag, into)) = decision.tag.checked_sub(1).and_then(|i| batch.get(i)) else {
                continue;
            };
            if !answered.insert(tag.id) {
                continue;
            }
            // Only a tag it was offered.
            let chosen = normalize_name(&decision.into)
                .and_then(|name| into.iter().find(|t| t.name == name))
                .filter(|t| {
                    db.tag_decision(&fold_key(&tag.name, &t.name))
                        .ok()
                        .flatten()
                        .is_none()
                });
            match chosen {
                Some(target) => {
                    folds.insert(tag.name.clone(), target.name.clone());
                }
                None => {
                    let names: Vec<&str> = into.iter().map(|t| t.name.as_str()).collect();
                    db.set_tag_decision(&keep_key(&tag.name), &format!("keep:{}", names.join(",")), "llm")?;
                }
            }
        }
    }
    let by_name: HashMap<&str, &TagInfo> = tags.iter().map(|t| (t.name.as_str(), t)).collect();
    out.proposed = group(&by_name, resolve_chains(folds));
    Ok(out)
}

/// Follows folds into tags that are folded themselves, so each tag goes to
/// its final tag; tags in a cycle are kept.
fn resolve_chains(folds: BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for from in folds.keys() {
        let mut into = &folds[from];
        let mut seen = HashSet::from([from]);
        let mut cycle = false;
        while let Some(next) = folds.get(into) {
            if !seen.insert(into) {
                cycle = true;
                break;
            }
            into = next;
        }
        if !cycle && into != from {
            out.insert(from.clone(), into.clone());
        }
    }
    out
}

/// One proposal per broader tag.
fn group(by_name: &HashMap<&str, &TagInfo>, folds: BTreeMap<String, String>) -> Vec<Proposed> {
    let mut groups: BTreeMap<&str, Vec<&TagInfo>> = BTreeMap::new();
    for (from, into) in &folds {
        if let (Some(from), Some(_)) = (by_name.get(from.as_str()), by_name.get(into.as_str())) {
            groups.entry(into.as_str()).or_default().push(from);
        }
    }
    groups
        .into_iter()
        .map(|(into, mut from)| {
            let into = by_name[into];
            from.sort_by(|a, b| b.pages.cmp(&a.pages).then(a.name.cmp(&b.name)));
            let mut names = from
                .iter()
                .take(NAMES_SHOWN)
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            if from.len() > NAMES_SHOWN {
                names.push_str(&format!(" and {} more", from.len() - NAMES_SHOWN));
            }
            Proposed {
                description: format!("fold   {names} → {} ({})", into.name, pages(into.pages)),
                decision_keys: from.iter().map(|t| fold_key(&t.name, &into.name)).collect(),
                change: TagChange::Fold {
                    into: into.id,
                    from: from.iter().map(|t| t.id).collect(),
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconcile::tests::{FakeReviewer, archive, no_names, tag_id};

    #[test]
    fn chains_end_in_one_tag_and_cycles_are_kept() {
        let folds = BTreeMap::from([
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
            ("x".to_string(), "y".to_string()),
            ("y".to_string(), "x".to_string()),
        ]);
        assert_eq!(
            resolve_chains(folds),
            BTreeMap::from([
                ("a".to_string(), "c".to_string()),
                ("b".to_string(), "c".to_string())
            ])
        );
    }

    #[tokio::test]
    async fn folds_only_into_tags_with_alike_pages() {
        let db = archive(&[
            // A redundant narrow tag on a python-programming page.
            (&["python-programming", "python-releases"], [1.0, 0.0, 0.0]),
            (&["python-programming"], [1.0, 0.1, 0.0]),
            // The snake: a different meaning, its pages unlike the language's.
            (&["python-snake"], [0.0, 1.0, 0.0]),
            (&["glacier"], [0.0, 0.0, 1.0]),
        ]);
        // Only python-releases has a candidate, so only it is asked about.
        let reviewer = FakeReviewer::new(vec![json!({"decisions": [
            {"tag": 1, "into": "Python Programming"}
        ]})]);
        let out = consolidate(&db, &reviewer, &no_names(), 0.85, |_| {})
            .await
            .unwrap();
        assert_eq!(out.checked, 1);
        assert!(reviewer.prompts.borrow()[0].contains("could fold into python-programming"));
        let descriptions: Vec<&str> = out.proposed.iter().map(|p| p.description.as_str()).collect();
        assert_eq!(
            descriptions,
            ["fold   python-releases → python-programming (2 pages)"]
        );
        assert_eq!(
            out.proposed[0].change,
            TagChange::Fold {
                into: tag_id(&db, "python-programming"),
                from: vec![tag_id(&db, "python-releases")]
            }
        );
    }

    #[tokio::test]
    async fn a_kept_tag_is_asked_again_only_for_new_candidates() {
        let db = archive(&[
            (&["coffee", "espresso-brewing"], [1.0, 0.0, 0.0]),
            (&["coffee"], [1.0, 0.1, 0.0]),
        ]);
        let reviewer = FakeReviewer::new(vec![json!({"decisions": [{"tag": 1, "into": ""}]})]);
        let out = consolidate(&db, &reviewer, &no_names(), 0.85, |_| {})
            .await
            .unwrap();
        assert!(out.proposed.is_empty());
        assert_eq!(
            db.tag_decision("consolidate:espresso-brewing")
                .unwrap()
                .as_deref(),
            Some("keep:coffee")
        );
        // The same candidate again: not asked (the fake would panic).
        let none = FakeReviewer::new(vec![]);
        let out = consolidate(&db, &none, &no_names(), 0.85, |_| {}).await.unwrap();
        assert_eq!(out.checked, 0);
    }

    #[tokio::test]
    async fn applying_a_fold_merges_and_undo_restores() {
        let mut db = archive(&[
            (&["sourdough"], [1.0, 0.0, 0.0]),
            (&["sourdough-starter"], [1.0, 0.1, 0.0]),
        ]);
        let (bread, starter) = (tag_id(&db, "sourdough"), tag_id(&db, "sourdough-starter"));
        db.apply_tag_changes(&[TagChange::Fold {
            into: bread,
            from: vec![starter],
        }])
        .unwrap();
        assert_eq!(db.find_tag("sourdough-starter").unwrap(), Some((bread, true)));
        db.undo_last_revision().unwrap();
        assert_eq!(db.find_tag("sourdough-starter").unwrap(), Some((starter, false)));
    }
}

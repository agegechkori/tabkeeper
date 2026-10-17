//! Whether a one-word tag names unrelated things (python the language and the
//! snake) or one thing with many sides (technology for phones and cars).
//!
//! A small model answering this in one go is close to a coin flip, so the
//! question is split up. Two designs answer it independently, and the word
//! counts as having several meanings only when both say so:
//! - staged: one short call per group of pages says what the word refers to
//!   there, then one call merges the phrases that refer to the same thing;
//! - reasoning first: one call writes what the word refers to in each group
//!   before it gives its verdict.
//!
//! On a labeled set of 20 words (scripts/meanings_bench.py) this was right 34
//! times out of 40 and missed no word with several meanings, at about 6 s a
//! word with qwen3:4b; the earlier single prompt was right 20 times without
//! thinking, and with thinking often ran out of tokens.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::db::AnalyzedPage;
use crate::llm::Reviewer;
use crate::tags::normalize_name;

/// Pages shown per group.
pub const PAGES_PER_GROUP: usize = 5;

const REFERENT: &str = "In one short phrase, say what the word \"{word}\" refers to in these web pages. Name the thing the word itself stands for, not the topic of each page. If the pages are about different examples or parts of one general thing the word names, name that general thing (for \"furniture\": \"furniture in general\"; for \"vehicles\": \"vehicles in general\"). Examples: \"the flying mammal\", \"the baseball club\", \"the computer device\", \"clothing in general\".

Reply with a JSON object: {\"referent\": \"...\"}.";

const COMPARE: &str = "A tag of a personal archive of web pages, \"{word}\", was found on several groups of pages. For each group you get a short phrase saying what the word refers to there.

Give each group a tag for the thing its phrase refers to. Groups that refer to the same thing get the same tag; different things get different tags, each qualified by its meaning (bat-animal, bat-baseball). Groups are the same thing when one is a kind, part or aspect of the other, or both are kinds of what the word names: chairs, tables and beds are all furniture; cars, bikes and ships are all vehicles; painting and sculpture are both art. Groups are different things only when nothing connects them except the word: bat the flying mammal and bat the baseball club; seal the animal and seal the stamp on a document; mouse the animal and mouse the computer device.

Reply with a JSON object with one entry per group, in order: {\"groups\": [{\"group\": 1, \"tag\": \"lowercase-kebab-case-name\", \"description\": \"one line\"}]}.";

const REASONED: &str = "A tag of a personal archive of web pages is used on the groups of pages below. Decide whether the word has one meaning in all groups or names unrelated things.

- First, for each group, write refers_to: what the word itself refers to there, in a short phrase. A meaning is what the word itself refers to, not the subject of the pages: furniture pages about chairs, tables and beds all use furniture in one meaning; art pages about painting and sculpture all use art in one meaning.
- Then set same_meaning. It is false only when the word names things that nothing connects except the word: bat the flying mammal and bat the baseball club; seal the animal and seal the stamp on a document; mouse the animal and mouse the computer device. Kinds, parts or aspects of one thing are one meaning: cars, bikes and ships are all vehicles.

Reply with a JSON object: {\"groups\": [{\"group\": 1, \"refers_to\": \"...\"}], \"same_meaning\": true}.";

#[derive(Deserialize)]
struct Referent {
    #[serde(default)]
    referent: String,
}

#[derive(Deserialize)]
struct Compared {
    groups: Vec<GroupThing>,
}

#[derive(Deserialize)]
struct GroupThing {
    group: usize,
    #[serde(default)]
    tag: String,
    #[serde(default)]
    description: String,
}

#[derive(Deserialize)]
struct Reasoned {
    same_meaning: bool,
}

/// What the word means across the groups.
#[derive(Debug, PartialEq)]
pub enum Verdict {
    Same,
    /// The distinct things, each as (tag, description, indexes of its groups).
    Several(Vec<(String, String, Vec<usize>)>),
}

fn first_sentence(text: &str) -> String {
    let text = text.trim();
    match text.find(". ") {
        Some(cut) if cut < 160 => text[..=cut].to_string(),
        _ => crate::extract::truncate_chars(text, 160),
    }
}

fn page_lines(pages: &[&AnalyzedPage]) -> String {
    pages
        .iter()
        .take(PAGES_PER_GROUP)
        .map(|p| format!("- {}: {}\n", p.title, first_sentence(&p.summary)))
        .collect()
}

async fn ask<R: Reviewer>(
    reviewer: &R,
    system: &str,
    user: &str,
    name: &str,
    schema: Value,
) -> Result<Value, String> {
    reviewer
        .ask_json(system, user, name, schema)
        .await
        .map_err(|err| format!("{err:#}"))
}

/// Asks whether `word` means the same thing in all `groups` of pages.
pub async fn check<R: Reviewer>(
    reviewer: &R,
    word: &str,
    groups: &[Vec<&AnalyzedPage>],
) -> Result<Verdict, String> {
    // Reasoning first, in one call.
    let mut user = format!("Tag: {word}\n");
    for (g, group) in groups.iter().enumerate() {
        user.push_str(&format!("\nGroup {}:\n{}", g + 1, page_lines(group)));
    }
    let schema = json!({
        "type": "object", "additionalProperties": false, "required": ["groups", "same_meaning"],
        "properties": {
            "groups": {"type": "array", "items": {"type": "object", "additionalProperties": false,
                "required": ["group", "refers_to"],
                "properties": {"group": {"type": "integer"}, "refers_to": {"type": "string"}}}},
            "same_meaning": {"type": "boolean"}
        }
    });
    let reasoned: Reasoned =
        serde_json::from_value(ask(reviewer, REASONED, &user, "tag_meanings", schema).await?)
            .map_err(|err| format!("the meanings reply was unusable: {err}"))?;
    if reasoned.same_meaning {
        return Ok(Verdict::Same);
    }

    // Staged: what the word refers to in each group, then which are the same.
    let referent_schema = json!({
        "type": "object", "additionalProperties": false, "required": ["referent"],
        "properties": {"referent": {"type": "string"}}
    });
    let system = REFERENT.replace("{word}", word);
    let mut referents = Vec::with_capacity(groups.len());
    for group in groups {
        let user = format!("Tag: {word}\nPages:\n{}", page_lines(group));
        let reply = ask(reviewer, &system, &user, "tag_referent", referent_schema.clone()).await?;
        let referent: Referent =
            serde_json::from_value(reply).map_err(|err| format!("the referent reply was unusable: {err}"))?;
        referents.push(referent.referent.trim().to_string());
    }
    let user: String = referents
        .iter()
        .enumerate()
        .map(|(i, r)| format!("{}. {r}\n", i + 1))
        .collect();
    let compare_schema = json!({
        "type": "object", "additionalProperties": false, "required": ["groups"],
        "properties": {"groups": {"type": "array", "items": {"type": "object", "additionalProperties": false,
            "required": ["group", "tag", "description"],
            "properties": {"group": {"type": "integer"}, "tag": {"type": "string"}, "description": {"type": "string"}}}}}
    });
    let raw = ask(
        reviewer,
        &COMPARE.replace("{word}", word),
        &user,
        "tag_things",
        compare_schema,
    )
    .await?;
    let compared: Compared =
        serde_json::from_value(raw).map_err(|err| format!("the comparison reply was unusable: {err}"))?;
    Ok(decide(word, groups.len(), &compared))
}

/// Several meanings only if the comparison gives every group a tag and the
/// groups get two or more different tags; anything less counts as one
/// meaning.
fn decide(word: &str, groups: usize, compared: &Compared) -> Verdict {
    let mut tag_of: Vec<Option<(String, String)>> = vec![None; groups];
    for g in &compared.groups {
        if let Some(slot) = g.group.checked_sub(1).and_then(|i| tag_of.get_mut(i)) {
            let name = normalize_name(&g.tag).unwrap_or_else(|| word.to_string());
            slot.get_or_insert((name, g.description.trim().to_string()));
        }
    }
    let mut things: BTreeMap<String, (String, Vec<usize>)> = BTreeMap::new();
    for (i, slot) in tag_of.into_iter().enumerate() {
        let Some((name, description)) = slot else {
            return Verdict::Same;
        };
        things.entry(name).or_insert((description, Vec::new())).1.push(i);
    }
    if things.len() < 2 {
        return Verdict::Same;
    }
    Verdict::Several(
        things
            .into_iter()
            .map(|(name, (description, members))| (name, description, members))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconcile::tests::FakeReviewer;

    fn page(title: &str) -> AnalyzedPage {
        AnalyzedPage {
            id: 0,
            title: title.into(),
            summary: "First sentence. Second sentence.".into(),
            tags: vec![],
        }
    }

    #[tokio::test]
    async fn one_meaning_when_the_reasoned_call_says_so() {
        let reviewer = FakeReviewer::new(vec![json!({"groups": [], "same_meaning": true})]);
        let (a, b) = (page("Bread"), page("Pear"));
        let verdict = check(&reviewer, "food", &[vec![&a], vec![&b]]).await.unwrap();
        assert_eq!(verdict, Verdict::Same);
        assert!(reviewer.prompts.borrow()[0].contains("- Bread: First sentence."));
    }

    #[tokio::test]
    async fn several_meanings_only_when_both_designs_agree() {
        let (lang, snake) = (page("Python tutorial"), page("Python (genus)"));
        let groups = [vec![&lang], vec![&snake]];
        let several = || json!({"groups": [], "same_meaning": false});
        // The staged design sees one thing: one meaning after all.
        let reviewer = FakeReviewer::new(vec![
            several(),
            json!({"referent": "the language"}),
            json!({"referent": "the language"}),
            json!({"groups": [{"group": 1, "tag": "python-programming", "description": "d"},
                              {"group": 2, "tag": "python-programming", "description": "d"}]}),
        ]);
        assert_eq!(check(&reviewer, "python", &groups).await.unwrap(), Verdict::Same);
        // A group left out: can't tell, so one meaning.
        let reviewer = FakeReviewer::new(vec![
            several(),
            json!({"referent": "the programming language"}),
            json!({"referent": "the snake"}),
            json!({"groups": [{"group": 1, "tag": "python-programming", "description": "d"}]}),
        ]);
        assert_eq!(check(&reviewer, "python", &groups).await.unwrap(), Verdict::Same);
        // Both see two things.
        let reviewer = FakeReviewer::new(vec![
            several(),
            json!({"referent": "the programming language"}),
            json!({"referent": "the snake"}),
            json!({"groups": [{"group": 1, "tag": "Python Programming", "description": "The language."},
                              {"group": 2, "tag": "python-snake", "description": "The snakes."}]}),
        ]);
        assert_eq!(
            check(&reviewer, "python", &groups).await.unwrap(),
            Verdict::Several(vec![
                ("python-programming".into(), "The language.".into(), vec![0]),
                ("python-snake".into(), "The snakes.".into(), vec![1]),
            ])
        );
    }
}

//! Storage for the tag review: applying merges, renames and splits as one
//! revision, and undoing the latest one.

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

use super::{Db, now};

/// The outcome of applying tag changes.
#[derive(Debug)]
pub struct Applied {
    /// The revision `undo` reverts, if any change was applied.
    pub revision: Option<i64>,
    /// Changes that couldn't be applied, by index, with the reason.
    pub failed: Vec<(usize, String)>,
}

/// One change to the tags, as proposed by the review.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum TagChange {
    /// Every page tagged `from` gets `into`; `from` becomes an alias of `into`.
    Merge { from: i64, into: i64 },
    /// The tag gets a new name; the old name becomes an alias.
    Rename { tag: i64, name: String },
    /// The tag covered several meanings, or pages it doesn't describe: each
    /// part gets its pages. A part with the tag's own name keeps the tag on its
    /// pages; one with another existing tag's name goes to that tag.
    Split { tag: i64, into: Vec<SplitPart> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitPart {
    pub name: String,
    pub description: String,
    pub pages: Vec<i64>,
}

/// A topic tag with the number of pages it's on.
#[derive(Debug, Clone, PartialEq)]
pub struct TagInfo {
    pub id: i64,
    pub name: String,
    pub description: Option<String>,
    pub locked: bool,
    pub pages: usize,
}

/// What applying a change did, recorded so it can be undone.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "step", rename_all = "snake_case")]
enum Step {
    /// These (page, raw tag) rows were moved from `from` to `to`.
    Retargeted {
        rows: Vec<(i64, String)>,
        from: i64,
        to: i64,
    },
    /// These aliases were moved from `from` to another tag.
    AliasesMoved {
        aliases: Vec<String>,
        from: i64,
    },
    AliasAdded {
        alias: String,
    },
    AliasesDeleted {
        aliases: Vec<(String, i64, String, bool, String)>,
    },
    /// A new tag; if pages got it after the revision, undoing moves them to `fallback`.
    TagCreated {
        id: i64,
        fallback: i64,
    },
    TagDeleted {
        id: i64,
        name: String,
        description: Option<String>,
        parent_id: Option<i64>,
        locked: bool,
        created_at: String,
    },
    Renamed {
        id: i64,
        old_name: String,
    },
}

impl Db {
    /// Topic tags (not `status/` ones) on at least one page, most used first.
    pub fn topic_tags(&self) -> Result<Vec<TagInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT t.id, t.name, t.description, t.locked, COUNT(DISTINCT pt.page_id) AS pages
             FROM tags t JOIN page_tags pt ON pt.resolved_tag_id = t.id JOIN pages p ON p.id = pt.page_id
             WHERE p.status = 'done' AND t.name NOT LIKE 'status/%'
             GROUP BY t.id ORDER BY pages DESC, t.name",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(TagInfo {
                id: r.get(0)?,
                name: r.get(1)?,
                description: r.get(2)?,
                locked: r.get(3)?,
                pages: usize::try_from(r.get::<_, i64>(4)?).unwrap_or(0),
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// (page id, title, summary) of the done pages with this tag.
    pub fn tag_pages(&self, tag_id: i64) -> Result<Vec<(i64, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT p.id, ifnull(p.title, p.url), ifnull(p.summary, '') FROM pages p
             JOIN page_tags pt ON pt.page_id = p.id
             WHERE pt.resolved_tag_id = ?1 AND p.status = 'done' ORDER BY p.id",
        )?;
        let rows = stmt.query_map([tag_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// The tag that uses `name`, as its name or as an alias.
    pub fn tag_named(&self, name: &str) -> Result<Option<i64>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id FROM tags WHERE name = ?1 UNION ALL SELECT tag_id FROM tag_aliases WHERE alias = ?1 LIMIT 1",
                [name],
                |r| r.get(0),
            )
            .optional()?)
    }

    pub fn tag_decision(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT decision FROM tag_decisions WHERE key = ?1", [key], |r| {
                r.get(0)
            })
            .optional()?)
    }

    pub fn set_tag_decision(&self, key: &str, decision: &str, source: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO tag_decisions (key, decision, source, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![key, decision, source, now()],
        )?;
        Ok(())
    }

    #[cfg(test)]
    pub fn apply_tag_changes(&mut self, changes: &[TagChange]) -> Result<Applied> {
        self.apply_reviewed_changes(changes, &[])
    }

    /// Applies the changes as one revision. A change that fails is left out,
    /// and the others are still applied. `decision_keys` holds each change's
    /// tag_decisions keys; undoing the revision declines those changes.
    pub fn apply_reviewed_changes(
        &mut self,
        changes: &[TagChange],
        decision_keys: &[Vec<String>],
    ) -> Result<Applied> {
        let mut tx = self.conn.transaction()?;
        let mut steps = Vec::new();
        let mut applied = Vec::new();
        let mut keys = Vec::new();
        let mut failed = Vec::new();
        for (i, change) in changes.iter().enumerate() {
            let sp = tx.savepoint()?;
            let mut change_steps = Vec::new();
            let result = match change {
                TagChange::Merge { from, into } => merge(&sp, *from, *into, &mut change_steps),
                TagChange::Rename { tag, name } => rename(&sp, *tag, name, &mut change_steps),
                TagChange::Split { tag, into } => split(&sp, *tag, into, &mut change_steps),
            };
            match result {
                Ok(()) => {
                    sp.commit()?;
                    steps.extend(change_steps);
                    applied.push(change.clone());
                    keys.extend(decision_keys.get(i).into_iter().flatten().cloned());
                }
                // Dropping the savepoint rolls this change back.
                Err(err) => failed.push((i, format!("{err:#}"))),
            }
        }
        let revision = if applied.is_empty() {
            None
        } else {
            tx.execute(
                "INSERT INTO revisions (created_at, status, changes, undo, decision_keys)
                 VALUES (?1, 'applied', ?2, ?3, ?4)",
                params![
                    now(),
                    serde_json::to_string(&applied)?,
                    serde_json::to_string(&steps)?,
                    serde_json::to_string(&keys)?
                ],
            )?;
            Some(tx.last_insert_rowid())
        };
        tx.commit()?;
        Ok(Applied { revision, failed })
    }

    /// Undoes the latest applied revision, and records its changes as
    /// declined so the review doesn't propose them again. Returns its id and
    /// how many changes it had, or `None` if there is nothing to undo.
    pub fn undo_last_revision(&mut self) -> Result<Option<(i64, usize)>> {
        let tx = self.conn.transaction()?;
        let latest: Option<(i64, String, String, String)> = tx
            .query_row(
                "SELECT id, changes, undo, decision_keys FROM revisions WHERE status = 'applied' ORDER BY id DESC LIMIT 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()?;
        let Some((id, changes, undo, keys)) = latest else {
            return Ok(None);
        };
        let changes: Vec<TagChange> = serde_json::from_str(&changes)?;
        let steps: Vec<Step> = serde_json::from_str(&undo).context("reading the revision's undo record")?;
        let mut freed_aliases = Vec::new();
        for step in steps.into_iter().rev() {
            undo_step(&tx, step, &mut freed_aliases)?;
        }
        return_alias_pages(&tx, &freed_aliases)?;
        let keys: Vec<String> = serde_json::from_str(&keys)?;
        for key in keys {
            tx.execute(
                "INSERT OR REPLACE INTO tag_decisions (key, decision, source, created_at) VALUES (?1, 'declined', 'user', ?2)",
                params![key, now()],
            )?;
        }
        tx.execute("UPDATE revisions SET status = 'undone' WHERE id = ?1", [id])?;
        tx.commit()?;
        Ok(Some((id, changes.len())))
    }
}

fn retarget(tx: &Connection, from: i64, to: i64, page: Option<i64>, steps: &mut Vec<Step>) -> Result<()> {
    let mut stmt = tx.prepare(
        "SELECT page_id, raw_tag FROM page_tags WHERE resolved_tag_id = ?1 AND (?2 IS NULL OR page_id = ?2)",
    )?;
    let rows: Vec<(i64, String)> = stmt
        .query_map(params![from, page], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (page_id, raw) in &rows {
        tx.execute(
            "UPDATE page_tags SET resolved_tag_id = ?3 WHERE page_id = ?1 AND raw_tag = ?2",
            params![page_id, raw, to],
        )?;
    }
    if !rows.is_empty() {
        steps.push(Step::Retargeted { rows, from, to });
    }
    Ok(())
}

fn delete_tag(tx: &Connection, id: i64, steps: &mut Vec<Step>) -> Result<()> {
    let (name, description, parent_id, locked, created_at) = tx.query_row(
        "SELECT name, description, parent_id, locked, created_at FROM tags WHERE id = ?1",
        [id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
    )?;
    tx.execute("DELETE FROM embeddings WHERE kind = 'tag' AND ref_id = ?1", [id])?;
    tx.execute("DELETE FROM tags WHERE id = ?1", [id])?;
    steps.push(Step::TagDeleted {
        id,
        name,
        description,
        parent_id,
        locked,
        created_at,
    });
    Ok(())
}

fn tag_name(tx: &Connection, id: i64) -> Result<String> {
    tx.query_row("SELECT name FROM tags WHERE id = ?1", [id], |r| r.get(0))
        .with_context(|| format!("tag {id} no longer exists"))
}

fn add_alias(tx: &Connection, alias: &str, tag: i64, steps: &mut Vec<Step>) -> Result<()> {
    let added = tx.execute(
        "INSERT OR IGNORE INTO tag_aliases (alias, tag_id, source, created_at) VALUES (?1, ?2, 'llm', ?3)",
        params![alias, tag, now()],
    )?;
    if added == 1 {
        steps.push(Step::AliasAdded {
            alias: alias.to_string(),
        });
    }
    Ok(())
}

fn merge(tx: &Connection, from: i64, into: i64, steps: &mut Vec<Step>) -> Result<()> {
    if from == into {
        bail!("can't merge a tag into itself");
    }
    let from_name = tag_name(tx, from)?;
    tag_name(tx, into)?;
    retarget(tx, from, into, None, steps)?;
    let aliases: Vec<String> = tx
        .prepare("SELECT alias FROM tag_aliases WHERE tag_id = ?1")?
        .query_map([from], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    if !aliases.is_empty() {
        tx.execute(
            "UPDATE tag_aliases SET tag_id = ?2 WHERE tag_id = ?1",
            params![from, into],
        )?;
        steps.push(Step::AliasesMoved { aliases, from });
    }
    delete_tag(tx, from, steps)?;
    add_alias(tx, &from_name, into, steps)
}

fn rename(tx: &Connection, tag: i64, name: &str, steps: &mut Vec<Step>) -> Result<()> {
    let old_name = tag_name(tx, tag)?;
    tx.execute("UPDATE tags SET name = ?2 WHERE id = ?1", params![tag, name])
        .with_context(|| format!("renaming {old_name} to {name}"))?;
    // The old embedding describes the old name; the next run makes a new one.
    tx.execute("DELETE FROM embeddings WHERE kind = 'tag' AND ref_id = ?1", [tag])?;
    steps.push(Step::Renamed {
        id: tag,
        old_name: old_name.clone(),
    });
    add_alias(tx, &old_name, tag, steps)
}

fn split(tx: &Connection, tag: i64, parts: &[SplitPart], steps: &mut Vec<Step>) -> Result<()> {
    let own_name = tag_name(tx, tag)?;
    let mut kept = 0;
    for part in parts {
        // A part with the tag's own name keeps the tag on the pages it fits.
        if part.name == own_name {
            kept += part.pages.len();
            continue;
        }
        // An existing tag, or the tag an alias with this name stands for.
        let existing: Option<i64> = tx
            .query_row(
                "SELECT id FROM tags WHERE name = ?1
                 UNION ALL SELECT tag_id FROM tag_aliases WHERE alias = ?1 AND tag_id != ?2
                 LIMIT 1",
                params![part.name, tag],
                |r| r.get(0),
            )
            .optional()?;
        let target = match existing {
            Some(id) => id,
            None => {
                // An alias of the split tag with this name would stand for
                // the old tag; the name now belongs to the new one.
                let own_alias: Option<(String, i64, String, bool, String)> = tx
                    .query_row(
                        "SELECT alias, tag_id, source, locked, created_at FROM tag_aliases WHERE alias = ?1",
                        [&part.name],
                        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
                    )
                    .optional()?;
                if let Some(alias) = own_alias {
                    tx.execute("DELETE FROM tag_aliases WHERE alias = ?1", [&part.name])?;
                    steps.push(Step::AliasesDeleted { aliases: vec![alias] });
                }
                tx.execute(
                    "INSERT INTO tags (name, description, created_at) VALUES (?1, ?2, ?3)",
                    params![part.name, part.description, now()],
                )?;
                let id = tx.last_insert_rowid();
                steps.push(Step::TagCreated { id, fallback: tag });
                id
            }
        };
        for page in &part.pages {
            retarget(tx, tag, target, Some(*page), steps)?;
        }
    }
    let remaining: i64 = tx.query_row(
        "SELECT COUNT(DISTINCT page_id) FROM page_tags WHERE resolved_tag_id = ?1",
        [tag],
        |r| r.get(0),
    )?;
    if remaining as usize != kept {
        bail!(
            "splitting {own_name} left {} pages without one of its parts",
            remaining as usize - kept.min(remaining as usize)
        );
    }
    if kept > 0 {
        // The tag lives on for the pages it fits, with its name and aliases.
        return Ok(());
    }
    // The tag's name is ambiguous now, so its aliases go with it.
    let aliases: Vec<(String, i64, String, bool, String)> = tx
        .prepare("SELECT alias, tag_id, source, locked, created_at FROM tag_aliases WHERE tag_id = ?1")?
        .query_map([tag], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
        })?
        .collect::<rusqlite::Result<_>>()?;
    if !aliases.is_empty() {
        tx.execute("DELETE FROM tag_aliases WHERE tag_id = ?1", [tag])?;
        steps.push(Step::AliasesDeleted { aliases });
    }
    delete_tag(tx, tag, steps)
}

/// Pages tagged after the revision reached the merged-into tag through an
/// alias the revision added or moved there. Once undone, those names are
/// tags (or aliases of tags) of their own again, and the pages follow them.
/// `freed` holds (alias, the tag it pointed to during the revision).
fn return_alias_pages(tx: &Connection, freed: &[(String, i64)]) -> Result<()> {
    for (alias, was) in freed {
        let now_on: Option<i64> = tx
            .query_row(
                "SELECT id FROM tags WHERE name = ?1 UNION ALL SELECT tag_id FROM tag_aliases WHERE alias = ?1 LIMIT 1",
                [alias],
                |r| r.get(0),
            )
            .optional()?;
        let Some(now_on) = now_on.filter(|id| id != was) else {
            continue;
        };
        let rows: Vec<(i64, String)> = tx
            .prepare("SELECT page_id, raw_tag FROM page_tags WHERE resolved_tag_id = ?1")?
            .query_map([was], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        for (page, raw) in rows {
            if crate::tags::normalize_name(&raw).as_deref() == Some(alias.as_str()) {
                tx.execute(
                    "UPDATE OR IGNORE page_tags SET resolved_tag_id = ?3 WHERE page_id = ?1 AND raw_tag = ?2",
                    params![page, raw, now_on],
                )?;
            }
        }
    }
    Ok(())
}

fn undo_step(tx: &Connection, step: Step, freed_aliases: &mut Vec<(String, i64)>) -> Result<()> {
    match step {
        Step::Retargeted { rows, from, to } => {
            for (page, raw) in rows {
                tx.execute(
                    "UPDATE page_tags SET resolved_tag_id = ?3 WHERE page_id = ?1 AND raw_tag = ?2 AND resolved_tag_id = ?4",
                    params![page, raw, from, to],
                )?;
            }
        }
        Step::AliasesMoved { aliases, from } => {
            for alias in aliases {
                let was: Option<i64> = tx
                    .query_row("SELECT tag_id FROM tag_aliases WHERE alias = ?1", [&alias], |r| {
                        r.get(0)
                    })
                    .optional()?;
                freed_aliases.extend(was.map(|was| (alias.clone(), was)));
                tx.execute(
                    "UPDATE tag_aliases SET tag_id = ?2 WHERE alias = ?1",
                    params![alias, from],
                )?;
            }
        }
        Step::AliasAdded { alias } => {
            let was: Option<i64> = tx
                .query_row("SELECT tag_id FROM tag_aliases WHERE alias = ?1", [&alias], |r| {
                    r.get(0)
                })
                .optional()?;
            tx.execute("DELETE FROM tag_aliases WHERE alias = ?1", [&alias])?;
            freed_aliases.extend(was.map(|was| (alias, was)));
        }
        Step::AliasesDeleted { aliases } => {
            for (alias, tag, source, locked, created_at) in aliases {
                tx.execute(
                    "INSERT OR REPLACE INTO tag_aliases (alias, tag_id, source, locked, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![alias, tag, source, locked, created_at],
                )?;
            }
        }
        Step::TagCreated { id, fallback } => {
            // Pages that got the new tag after the revision keep a tag.
            tx.execute(
                "UPDATE page_tags SET resolved_tag_id = ?2 WHERE resolved_tag_id = ?1",
                params![id, fallback],
            )?;
            tx.execute("DELETE FROM tag_aliases WHERE tag_id = ?1", [id])?;
            tx.execute("DELETE FROM embeddings WHERE kind = 'tag' AND ref_id = ?1", [id])?;
            tx.execute("DELETE FROM tags WHERE id = ?1", [id])?;
        }
        Step::TagDeleted {
            id,
            name,
            description,
            parent_id,
            locked,
            created_at,
        } => {
            // A later run may have created a tag with the freed name, for a
            // page whose raw tag is that name: it joins the restored tag.
            let newcomer: Option<i64> = tx
                .query_row("SELECT id FROM tags WHERE name = ?1", [&name], |r| r.get(0))
                .optional()?;
            if let Some(newcomer) = newcomer {
                tx.execute(
                    "UPDATE tags SET name = name || ' (undone)' WHERE id = ?1",
                    [newcomer],
                )?;
            }
            tx.execute(
                "INSERT INTO tags (id, name, description, parent_id, locked, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![id, name, description, parent_id, locked, created_at],
            )
            .with_context(|| format!("restoring tag {name}"))?;
            if let Some(newcomer) = newcomer {
                for sql in [
                    "UPDATE page_tags SET resolved_tag_id = ?2 WHERE resolved_tag_id = ?1",
                    "UPDATE tag_aliases SET tag_id = ?2 WHERE tag_id = ?1",
                    "UPDATE tags SET parent_id = ?2 WHERE parent_id = ?1",
                ] {
                    tx.execute(sql, params![newcomer, id])?;
                }
                tx.execute(
                    "DELETE FROM embeddings WHERE kind = 'tag' AND ref_id = ?1",
                    [newcomer],
                )?;
                tx.execute("DELETE FROM tags WHERE id = ?1", [newcomer])?;
            }
        }
        Step::Renamed { id, old_name } => {
            tx.execute("UPDATE tags SET name = ?2 WHERE id = ?1", params![id, old_name])?;
            tx.execute("DELETE FROM embeddings WHERE kind = 'tag' AND ref_id = ?1", [id])?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::PageResult;

    /// Pages a..e with the given (raw tag, tag) pairs each.
    fn archive(pages: &[&[&str]]) -> (Db, Vec<i64>) {
        let mut db = Db::open_in_memory().unwrap();
        let mut ids = Vec::new();
        for (i, tags) in pages.iter().enumerate() {
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
            ids.push(id);
        }
        (db, ids)
    }

    fn names(db: &Db) -> Vec<(String, usize)> {
        let mut tags: Vec<(String, usize)> = db
            .topic_tags()
            .unwrap()
            .into_iter()
            .map(|t| (t.name, t.pages))
            .collect();
        tags.sort();
        tags
    }

    fn id(db: &Db, name: &str) -> i64 {
        db.find_tag(name).unwrap().unwrap().0
    }

    #[test]
    fn merge_and_undo() {
        let (mut db, _) = archive(&[
            &["ml", "python"],
            &["machine-learning"],
            &["machine-learning", "ml"],
        ]);
        let (ml, full) = (id(&db, "ml"), id(&db, "machine-learning"));
        db.add_alias("m-l", ml, "rule").unwrap();
        let before = names(&db);

        db.apply_tag_changes(&[TagChange::Merge { from: ml, into: full }])
            .unwrap();
        assert_eq!(
            names(&db),
            [("machine-learning".to_string(), 3), ("python".to_string(), 1)]
        );
        assert_eq!(
            db.find_tag("ml").unwrap(),
            Some((full, true)),
            "the old name is an alias now"
        );
        assert_eq!(
            db.find_tag("m-l").unwrap(),
            Some((full, true)),
            "its aliases moved too"
        );

        assert_eq!(db.undo_last_revision().unwrap(), Some((1, 1)));
        assert_eq!(names(&db), before);
        assert_eq!(db.find_tag("ml").unwrap(), Some((ml, false)));
        assert_eq!(db.find_tag("m-l").unwrap(), Some((ml, true)));
        assert_eq!(db.undo_last_revision().unwrap(), None, "nothing left to undo");
    }

    #[test]
    fn undo_after_a_new_tag_takes_no_old_id() {
        let (mut db, _) = archive(&[&["board-game"], &["board-games"]]);
        let (one, many) = (id(&db, "board-game"), id(&db, "board-games"));
        db.apply_tag_changes(&[TagChange::Merge {
            from: many,
            into: one,
        }])
        .unwrap();
        // The newest tag was deleted; a new tag must not get its id.
        let new = db.create_tag("chess", None).unwrap();
        assert!(new > many);
        db.undo_last_revision().unwrap();
        assert_eq!(db.find_tag("board-games").unwrap(), Some((many, false)));
    }

    #[test]
    fn a_split_part_follows_a_merge_applied_before_it() {
        let (mut db, _) = archive(&[&["rust"], &["rust"], &["oxides"], &["oxide"]]);
        let (rust, plural, oxide) = (id(&db, "rust"), id(&db, "oxides"), id(&db, "oxide"));
        let pages = db.tag_pages(rust).unwrap();
        let part = |name: &str, page: i64| SplitPart {
            name: name.into(),
            description: String::new(),
            pages: vec![page],
        };
        db.apply_tag_changes(&[
            TagChange::Merge {
                from: plural,
                into: oxide,
            },
            TagChange::Split {
                tag: rust,
                into: vec![part("rust", pages[0].0), part("oxides", pages[1].0)],
            },
        ])
        .unwrap();
        assert_eq!(names(&db), [("oxide".to_string(), 3), ("rust".to_string(), 1)]);
    }

    #[test]
    fn undone_changes_are_declined() {
        let (mut db, _) = archive(&[&["ml"], &["machine-learning"]]);
        let (ml, full) = (id(&db, "ml"), id(&db, "machine-learning"));
        let key = "merge:machine-learning|ml".to_string();
        db.apply_reviewed_changes(&[TagChange::Merge { from: ml, into: full }], &[vec![key.clone()]])
            .unwrap();
        assert_eq!(db.tag_decision(&key).unwrap(), None);
        db.undo_last_revision().unwrap();
        assert_eq!(db.tag_decision(&key).unwrap().as_deref(), Some("declined"));
    }

    #[test]
    fn undoing_a_merge_returns_pages_tagged_since() {
        let (mut db, pages) = archive(&[&["ml"], &["machine-learning"]]);
        let (ml, full) = (id(&db, "ml"), id(&db, "machine-learning"));
        db.apply_tag_changes(&[TagChange::Merge { from: ml, into: full }])
            .unwrap();
        // A later run resolves a page's raw tag "ML" through the new alias.
        db.conn
            .execute(
                "INSERT INTO page_tags (page_id, raw_tag, resolved_tag_id) VALUES (?1, 'ML', ?2)",
                params![pages[1], full],
            )
            .unwrap();
        db.undo_last_revision().unwrap();
        assert_eq!(
            names(&db),
            [("machine-learning".to_string(), 1), ("ml".to_string(), 2)]
        );
    }

    #[test]
    fn a_split_part_named_like_the_tags_own_alias_takes_the_name() {
        let (mut db, pages) = archive(&[&["go"], &["go"]]);
        let go = id(&db, "go");
        db.add_alias("golang", go, "rule").unwrap();
        let part = |name: &str, page: i64| SplitPart {
            name: name.into(),
            description: String::new(),
            pages: vec![page],
        };
        db.apply_tag_changes(&[TagChange::Split {
            tag: go,
            into: vec![part("go", pages[0]), part("golang", pages[1])],
        }])
        .unwrap();
        let golang = db.find_tag("golang").unwrap().unwrap();
        assert!(
            golang.0 != go && !golang.1,
            "golang is its own tag, not an alias: {golang:?}"
        );
        db.undo_last_revision().unwrap();
        assert_eq!(db.find_tag("golang").unwrap(), Some((go, true)));
        assert_eq!(names(&db), [("go".to_string(), 2)]);
    }

    #[test]
    fn rename_and_undo() {
        let (mut db, _) = archive(&[&["js"]]);
        let js = id(&db, "js");
        db.apply_tag_changes(&[TagChange::Rename {
            tag: js,
            name: "javascript".into(),
        }])
        .unwrap();
        assert_eq!(names(&db), [("javascript".to_string(), 1)]);
        assert_eq!(db.find_tag("js").unwrap(), Some((js, true)));
        db.undo_last_revision().unwrap();
        assert_eq!(names(&db), [("js".to_string(), 1)]);
        assert_eq!(db.find_tag("javascript").unwrap(), None);
    }

    #[test]
    fn split_into_new_and_existing_tags_and_undo() {
        let (mut db, pages) = archive(&[&["rust"], &["rust", "systems"], &["rust"], &["rust-corrosion"]]);
        let rust = id(&db, "rust");
        let before = names(&db);
        let change = TagChange::Split {
            tag: rust,
            into: vec![
                SplitPart {
                    name: "rust-programming".into(),
                    description: "The language".into(),
                    pages: vec![pages[0], pages[1]],
                },
                SplitPart {
                    name: "rust-corrosion".into(),
                    description: "Iron oxide".into(),
                    pages: vec![pages[2]],
                },
            ],
        };
        db.apply_tag_changes(&[change]).unwrap();
        assert_eq!(
            names(&db),
            [
                ("rust-corrosion".to_string(), 2),
                ("rust-programming".to_string(), 2),
                ("systems".to_string(), 1)
            ]
        );
        assert_eq!(db.find_tag("rust").unwrap(), None);

        db.undo_last_revision().unwrap();
        assert_eq!(names(&db), before);
        assert_eq!(
            db.find_tag("rust-programming").unwrap(),
            None,
            "the new tag is gone"
        );
    }

    #[test]
    fn a_split_can_keep_the_tag_on_the_pages_it_fits() {
        let (mut db, pages) = archive(&[
            &["web-development"],
            &["web-development"],
            &["web-development", "rust"],
        ]);
        let web = id(&db, "web-development");
        let change = TagChange::Split {
            tag: web,
            into: vec![
                SplitPart {
                    name: "web-development".into(),
                    description: "".into(),
                    pages: vec![pages[0], pages[1]],
                },
                SplitPart {
                    name: "rust".into(),
                    description: "".into(),
                    pages: vec![pages[2]],
                },
            ],
        };
        db.apply_tag_changes(&[change]).unwrap();
        assert_eq!(
            names(&db),
            [("rust".to_string(), 1), ("web-development".to_string(), 2)]
        );
        db.undo_last_revision().unwrap();
        assert_eq!(
            names(&db),
            [("rust".to_string(), 1), ("web-development".to_string(), 3)]
        );
    }

    #[test]
    fn a_split_must_place_every_page() {
        let (mut db, pages) = archive(&[&["rust"], &["rust"]]);
        let rust = id(&db, "rust");
        let change = TagChange::Split {
            tag: rust,
            into: vec![SplitPart {
                name: "rust-programming".into(),
                description: "".into(),
                pages: vec![pages[0]],
            }],
        };
        let applied = db.apply_tag_changes(&[change]).unwrap();
        assert_eq!(applied.revision, None);
        assert_eq!(applied.failed.len(), 1);
        assert_eq!(names(&db), [("rust".to_string(), 2)], "nothing was changed");
    }

    #[test]
    fn a_failing_change_leaves_the_others_applied() {
        let (mut db, pages) = archive(&[&["rust"], &["rust"], &["ml"], &["machine-learning"]]);
        let (rust, ml, full) = (id(&db, "rust"), id(&db, "ml"), id(&db, "machine-learning"));
        let bad_split = TagChange::Split {
            tag: rust,
            into: vec![SplitPart {
                name: "rust-programming".into(),
                description: "".into(),
                pages: vec![pages[0]],
            }],
        };
        let applied = db
            .apply_tag_changes(&[bad_split, TagChange::Merge { from: ml, into: full }])
            .unwrap();
        assert_eq!(applied.failed.iter().map(|(i, _)| *i).collect::<Vec<_>>(), [0]);
        assert_eq!(
            names(&db),
            [("machine-learning".to_string(), 2), ("rust".to_string(), 2)]
        );
        // The revision holds only the merge.
        assert_eq!(
            db.undo_last_revision().unwrap(),
            Some((applied.revision.unwrap(), 1))
        );
        assert_eq!(names(&db).len(), 3);
    }

    #[test]
    fn undo_absorbs_a_tag_recreated_with_a_split_tags_name() {
        let (mut db, pages) = archive(&[&["rust"], &["rust"]]);
        let rust = id(&db, "rust");
        let part = |name: &str, page: i64| SplitPart {
            name: name.into(),
            description: String::new(),
            pages: vec![page],
        };
        db.apply_tag_changes(&[TagChange::Split {
            tag: rust,
            into: vec![
                part("rust-programming", pages[0]),
                part("rust-corrosion", pages[1]),
            ],
        }])
        .unwrap();
        assert_eq!(db.find_tag("rust").unwrap(), None, "the name is free");
        // A later run tags a page with the raw tag rust again.
        let newcomer = db.create_tag("rust", None).unwrap();
        db.conn
            .execute(
                "INSERT INTO page_tags (page_id, raw_tag, resolved_tag_id) VALUES (?1, 'rust-lang', ?2)",
                params![pages[0], newcomer],
            )
            .unwrap();
        db.undo_last_revision().unwrap();
        assert_eq!(db.find_tag("rust").unwrap(), Some((rust, false)));
        assert_eq!(names(&db), [("rust".to_string(), 2)]);
    }

    #[test]
    fn decisions_are_remembered() {
        let db = Db::open_in_memory().unwrap();
        assert_eq!(db.tag_decision("merge:a|b").unwrap(), None);
        db.set_tag_decision("merge:a|b", "separate", "llm").unwrap();
        assert_eq!(db.tag_decision("merge:a|b").unwrap().as_deref(), Some("separate"));
    }
}

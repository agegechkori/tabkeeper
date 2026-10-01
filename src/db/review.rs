//! Storage for the tag review: applying merges, renames and splits as one
//! revision, and undoing the latest one.

use anyhow::{Context, Result, bail};
use rusqlite::{OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};

use super::{Db, now};

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

    /// Applies the changes as one revision and returns its id.
    pub fn apply_tag_changes(&mut self, changes: &[TagChange]) -> Result<i64> {
        let tx = self.conn.transaction()?;
        let mut steps = Vec::new();
        for change in changes {
            match change {
                TagChange::Merge { from, into } => merge(&tx, *from, *into, &mut steps)?,
                TagChange::Rename { tag, name } => rename(&tx, *tag, name, &mut steps)?,
                TagChange::Split { tag, into } => split(&tx, *tag, into, &mut steps)?,
            }
        }
        tx.execute(
            "INSERT INTO revisions (created_at, status, changes, undo) VALUES (?1, 'applied', ?2, ?3)",
            params![
                now(),
                serde_json::to_string(changes)?,
                serde_json::to_string(&steps)?
            ],
        )?;
        let id = tx.last_insert_rowid();
        tx.commit()?;
        Ok(id)
    }

    /// Undoes the latest applied revision. Returns its id and how many changes
    /// it had, or `None` if there is nothing to undo.
    pub fn undo_last_revision(&mut self) -> Result<Option<(i64, usize)>> {
        let tx = self.conn.transaction()?;
        let latest: Option<(i64, String, String)> = tx
            .query_row(
                "SELECT id, changes, undo FROM revisions WHERE status = 'applied' ORDER BY id DESC LIMIT 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        let Some((id, changes, undo)) = latest else {
            return Ok(None);
        };
        let changes: Vec<TagChange> = serde_json::from_str(&changes)?;
        let steps: Vec<Step> = serde_json::from_str(&undo).context("reading the revision's undo record")?;
        for step in steps.into_iter().rev() {
            undo_step(&tx, step)?;
        }
        tx.execute("UPDATE revisions SET status = 'undone' WHERE id = ?1", [id])?;
        tx.commit()?;
        Ok(Some((id, changes.len())))
    }
}

fn retarget(tx: &Transaction, from: i64, to: i64, page: Option<i64>, steps: &mut Vec<Step>) -> Result<()> {
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

fn delete_tag(tx: &Transaction, id: i64, steps: &mut Vec<Step>) -> Result<()> {
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

fn tag_name(tx: &Transaction, id: i64) -> Result<String> {
    tx.query_row("SELECT name FROM tags WHERE id = ?1", [id], |r| r.get(0))
        .with_context(|| format!("tag {id} no longer exists"))
}

fn add_alias(tx: &Transaction, alias: &str, tag: i64, steps: &mut Vec<Step>) -> Result<()> {
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

fn merge(tx: &Transaction, from: i64, into: i64, steps: &mut Vec<Step>) -> Result<()> {
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

fn rename(tx: &Transaction, tag: i64, name: &str, steps: &mut Vec<Step>) -> Result<()> {
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

fn split(tx: &Transaction, tag: i64, parts: &[SplitPart], steps: &mut Vec<Step>) -> Result<()> {
    let own_name = tag_name(tx, tag)?;
    let mut kept = 0;
    for part in parts {
        // A part with the tag's own name keeps the tag on the pages it fits.
        if part.name == own_name {
            kept += part.pages.len();
            continue;
        }
        let existing: Option<i64> = tx
            .query_row("SELECT id FROM tags WHERE name = ?1", [&part.name], |r| r.get(0))
            .optional()?;
        let target = match existing {
            Some(id) => id,
            None => {
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

fn undo_step(tx: &Transaction, step: Step) -> Result<()> {
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
                tx.execute(
                    "UPDATE tag_aliases SET tag_id = ?2 WHERE alias = ?1",
                    params![alias, from],
                )?;
            }
        }
        Step::AliasAdded { alias } => {
            tx.execute("DELETE FROM tag_aliases WHERE alias = ?1", [alias])?;
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
            tx.execute(
                "INSERT INTO tags (id, name, description, parent_id, locked, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![id, name, description, parent_id, locked, created_at],
            )
            .with_context(|| format!("restoring tag {name}; a tag with that name was created since"))?;
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
        assert!(db.apply_tag_changes(&[change]).is_err());
        assert_eq!(names(&db), [("rust".to_string(), 2)], "nothing was changed");
    }

    #[test]
    fn decisions_are_remembered() {
        let db = Db::open_in_memory().unwrap();
        assert_eq!(db.tag_decision("merge:a|b").unwrap(), None);
        db.set_tag_decision("merge:a|b", "separate", "llm").unwrap();
        assert_eq!(db.tag_decision("merge:a|b").unwrap().as_deref(), Some("separate"));
    }
}

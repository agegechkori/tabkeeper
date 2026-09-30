use std::path::Path;

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};

use crate::tags::{TagPath, TagRow};

const SCHEMA_VERSION: i32 = 1;

const SCHEMA_V1: &str = "
CREATE TABLE pages (
    id            INTEGER PRIMARY KEY,
    url           TEXT NOT NULL UNIQUE,   -- normalized, used for deduplication
    original_url  TEXT NOT NULL,
    browser_title TEXT,
    source        TEXT NOT NULL,          -- where the URL came from, e.g. 'import'
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'done', 'failed')),
    error         TEXT,
    title         TEXT,
    summary       TEXT,
    lang          TEXT,
    note_file     TEXT UNIQUE,            -- file name under notes/, kept stable across renders
    added_at      TEXT NOT NULL,
    processed_at  TEXT
);

CREATE TABLE tags (
    id          INTEGER PRIMARY KEY,
    parent_id   INTEGER REFERENCES tags(id),
    name        TEXT NOT NULL,
    path        TEXT NOT NULL UNIQUE,
    description TEXT,
    locked      INTEGER NOT NULL DEFAULT 0
);

-- raw_tag is exactly what the model returned and is never modified; tag
-- revisions change how it resolves, not the raw value.
CREATE TABLE page_tags (
    page_id         INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    raw_tag         TEXT NOT NULL,
    resolved_tag_id INTEGER NOT NULL REFERENCES tags(id),
    PRIMARY KEY (page_id, raw_tag)
);
CREATE INDEX page_tags_by_tag ON page_tags(resolved_tag_id);
";

pub struct Db {
    conn: Connection,
}

#[derive(Debug, Clone)]
pub struct PendingPage {
    pub id: i64,
    pub url: String,
    pub browser_title: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DonePage {
    pub id: i64,
    pub url: String,
    pub source: String,
    pub title: String,
    pub summary: String,
    pub lang: Option<String>,
    pub note_file: Option<String>,
    pub processed_at: String,
}

pub struct PageResult<'a> {
    pub title: &'a str,
    pub summary: &'a str,
    pub lang: Option<&'a str>,
    /// (raw tag as returned by the model, resolved tag id)
    pub tags: &'a [(String, i64)],
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct StatusCounts {
    pub pending: usize,
    pub done: usize,
    pub failed: usize,
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        Self::init(conn)
    }

    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        let version: i32 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
        match version {
            0 => {
                conn.execute_batch(SCHEMA_V1)?;
                conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
            }
            SCHEMA_VERSION => {}
            v => {
                bail!("database schema version {v} is newer than this tabkeeper supports ({SCHEMA_VERSION})")
            }
        }
        Ok(Self { conn })
    }

    /// Adds a page to process. Returns false if the URL is already known.
    pub fn add_page(
        &self,
        url: &str,
        original_url: &str,
        browser_title: Option<&str>,
        source: &str,
    ) -> Result<bool> {
        let n = self.conn.execute(
            "INSERT OR IGNORE INTO pages (url, original_url, browser_title, source, added_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![url, original_url, browser_title, source, now()],
        )?;
        Ok(n == 1)
    }

    pub fn retry_failed(&self) -> Result<usize> {
        Ok(self.conn.execute(
            "UPDATE pages SET status = 'pending', error = NULL WHERE status = 'failed'",
            [],
        )?)
    }

    pub fn pending_pages(&self) -> Result<Vec<PendingPage>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, url, browser_title FROM pages WHERE status = 'pending' ORDER BY id")?;
        let rows = stmt.query_map([], |r| {
            Ok(PendingPage {
                id: r.get(0)?,
                url: r.get(1)?,
                browser_title: r.get(2)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn status_counts(&self) -> Result<StatusCounts> {
        let mut counts = StatusCounts::default();
        let mut stmt = self
            .conn
            .prepare("SELECT status, COUNT(*) FROM pages GROUP BY status")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
        for row in rows {
            let (status, n) = row?;
            let n = usize::try_from(n)?;
            match status.as_str() {
                "pending" => counts.pending = n,
                "done" => counts.done = n,
                _ => counts.failed = n,
            }
        }
        Ok(counts)
    }

    pub fn mark_failed(&self, page_id: i64, error: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE pages SET status = 'failed', error = ?2, processed_at = ?3 WHERE id = ?1",
            params![page_id, error, now()],
        )?;
        Ok(())
    }

    pub fn save_result(&mut self, page_id: i64, result: &PageResult) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "UPDATE pages SET status = 'done', error = NULL, title = ?2, summary = ?3, lang = ?4, processed_at = ?5
             WHERE id = ?1",
            params![page_id, result.title, result.summary, result.lang, now()],
        )?;
        tx.execute("DELETE FROM page_tags WHERE page_id = ?1", [page_id])?;
        for (raw, tag_id) in result.tags {
            tx.execute(
                "INSERT OR IGNORE INTO page_tags (page_id, raw_tag, resolved_tag_id) VALUES (?1, ?2, ?3)",
                params![page_id, raw, tag_id],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn tag_id(&self, path: &TagPath) -> Result<Option<i64>> {
        Ok(self
            .conn
            .query_row("SELECT id FROM tags WHERE path = ?1", [path.to_string()], |r| {
                r.get(0)
            })
            .optional()?)
    }

    /// Returns the id of `path`, creating it and any missing ancestors. The
    /// description is only set on a tag this call creates.
    pub fn ensure_tag(&self, path: &TagPath, description: Option<&str>) -> Result<i64> {
        let mut parent: Option<i64> = None;
        let mut id = 0;
        let depth = path.segments().len();
        for (i, prefix) in path.prefixes().enumerate() {
            id = match self.tag_id(&prefix)? {
                Some(existing) => existing,
                None => {
                    let desc = if i + 1 == depth { description } else { None };
                    self.conn.execute(
                        "INSERT INTO tags (parent_id, name, path, description) VALUES (?1, ?2, ?3, ?4)",
                        params![parent, prefix.segments()[i], prefix.to_string(), desc],
                    )?;
                    self.conn.last_insert_rowid()
                }
            };
            parent = Some(id);
        }
        Ok(id)
    }

    pub fn tags(&self) -> Result<Vec<TagRow>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, parent_id, name, path, description FROM tags ORDER BY path")?;
        let rows = stmt.query_map([], |r| {
            Ok(TagRow {
                id: r.get(0)?,
                parent_id: r.get(1)?,
                name: r.get(2)?,
                path: r.get(3)?,
                description: r.get(4)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Distinct (page id, tag id) pairs for pages that are done.
    pub fn tag_links(&self) -> Result<Vec<(i64, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT pt.page_id, pt.resolved_tag_id
             FROM page_tags pt JOIN pages p ON p.id = pt.page_id
             WHERE p.status = 'done'",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn done_pages(&self) -> Result<Vec<DonePage>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, url, source, title, summary, lang, note_file, processed_at
             FROM pages WHERE status = 'done' ORDER BY id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(DonePage {
                id: r.get(0)?,
                url: r.get(1)?,
                source: r.get(2)?,
                title: r.get(3)?,
                summary: r.get(4)?,
                lang: r.get(5)?,
                note_file: r.get(6)?,
                processed_at: r.get(7)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn set_note_file(&self, page_id: i64, file: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE pages SET note_file = ?2 WHERE id = ?1",
            params![page_id, file],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedupes_pages() {
        let db = Db::open_in_memory().unwrap();
        assert!(
            db.add_page("https://a.com/", "https://a.com/#x", None, "import")
                .unwrap()
        );
        assert!(
            !db.add_page("https://a.com/", "https://a.com/", None, "import")
                .unwrap()
        );
        assert_eq!(db.pending_pages().unwrap().len(), 1);
    }

    #[test]
    fn ensure_tag_creates_ancestors_once() {
        let db = Db::open_in_memory().unwrap();
        let rust = TagPath::parse("tech/languages/rust", 3).unwrap();
        let go = TagPath::parse("tech/languages/go", 3).unwrap();
        let rust_id = db.ensure_tag(&rust, Some("The Rust language")).unwrap();
        db.ensure_tag(&go, None).unwrap();
        assert_eq!(db.ensure_tag(&rust, Some("ignored")).unwrap(), rust_id);

        let tags = db.tags().unwrap();
        let paths: Vec<&str> = tags.iter().map(|t| t.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "tech",
                "tech/languages",
                "tech/languages/go",
                "tech/languages/rust"
            ]
        );
        let rust_row = tags.iter().find(|t| t.id == rust_id).unwrap();
        assert_eq!(rust_row.description.as_deref(), Some("The Rust language"));
        assert_eq!(rust_row.name, "rust");
        let languages = tags.iter().find(|t| t.path == "tech/languages").unwrap();
        assert_eq!(rust_row.parent_id, Some(languages.id));
    }

    #[test]
    fn save_result_replaces_tags() {
        let mut db = Db::open_in_memory().unwrap();
        db.add_page("https://a.com/", "https://a.com/", None, "import")
            .unwrap();
        let page = db.pending_pages().unwrap()[0].id;
        let a = db.ensure_tag(&TagPath::parse("a", 3).unwrap(), None).unwrap();
        let b = db.ensure_tag(&TagPath::parse("b", 3).unwrap(), None).unwrap();
        let first = [("A".to_string(), a)];
        db.save_result(
            page,
            &PageResult {
                title: "T",
                summary: "S",
                lang: Some("en"),
                tags: &first,
            },
        )
        .unwrap();
        let second = [("b".to_string(), b)];
        db.save_result(
            page,
            &PageResult {
                title: "T",
                summary: "S",
                lang: None,
                tags: &second,
            },
        )
        .unwrap();
        assert_eq!(db.tag_links().unwrap(), [(page, b)]);
        assert_eq!(
            db.status_counts().unwrap(),
            StatusCounts {
                pending: 0,
                done: 1,
                failed: 0
            }
        );
    }
}

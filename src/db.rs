use std::path::Path;

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};

use crate::tags::TagRow;

const SCHEMA_VERSION: i32 = 3;

const PAGES_TABLE: &str = "
CREATE TABLE pages (
    id            INTEGER PRIMARY KEY,
    url           TEXT NOT NULL UNIQUE,   -- normalized, used for deduplication
    original_url  TEXT NOT NULL,
    browser_title TEXT,
    source        TEXT NOT NULL,          -- where the URL came from, e.g. 'import'
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'done', 'failed', 'unreachable')),
    error         TEXT,
    error_kind    TEXT,                   -- why it failed or was unreachable, e.g. 'not_found'
    title         TEXT,
    summary       TEXT,
    lang          TEXT,
    note_file     TEXT UNIQUE,            -- file name under notes/, kept stable across renders
    added_at      TEXT NOT NULL,
    processed_at  TEXT
);
";

const SCHEMA: &str = "
-- Every tag has a unique flat name. parent_id places it in the tag tree,
-- which the end-of-run reconciliation builds; until then all tags are roots.
CREATE TABLE tags (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    parent_id   INTEGER REFERENCES tags(id),
    description TEXT,
    locked      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
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

-- Other names that resolve to a tag. source: rule (spelling variants merged
-- in code), llm (merges from reconciliation) or user.
CREATE TABLE tag_aliases (
    alias      TEXT PRIMARY KEY,
    tag_id     INTEGER NOT NULL REFERENCES tags(id),
    source     TEXT NOT NULL CHECK (source IN ('rule', 'llm', 'user')),
    locked     INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

-- Unit-length f32 vectors, little-endian. kind 'tag' refers to tags.id,
-- 'page' to pages.id (the page's title and summary).
CREATE TABLE embeddings (
    kind   TEXT NOT NULL CHECK (kind IN ('tag', 'page')),
    ref_id INTEGER NOT NULL,
    model  TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (kind, ref_id, model)
);
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
    pub unreachable: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingKind {
    Tag,
    Page,
}

impl EmbeddingKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Tag => "tag",
            Self::Page => "page",
        }
    }
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn vector_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn blob_to_vector(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
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
                conn.execute_batch(PAGES_TABLE)?;
                conn.execute_batch(SCHEMA)?;
                conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
            }
            2 => migrate_v2_to_v3(&conn)?,
            1 => bail!(
                "this output folder was created by an early development version of tabkeeper that used \
                 hierarchical tags per page; use a new --out folder"
            ),
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

    /// Makes failed and unreachable pages pending again.
    pub fn retry_failed(&self) -> Result<usize> {
        Ok(self.conn.execute(
            "UPDATE pages SET status = 'pending', error = NULL, error_kind = NULL
             WHERE status IN ('failed', 'unreachable')",
            [],
        )?)
    }

    /// Makes these pages pending again, e.g. after pages were failed for what
    /// turned out to be a problem with the model rather than with them.
    pub fn reset_to_pending(&self, page_ids: &[i64]) -> Result<()> {
        for id in page_ids {
            self.conn.execute(
                "UPDATE pages SET status = 'pending', error = NULL, error_kind = NULL WHERE id = ?1",
                [id],
            )?;
        }
        Ok(())
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
                "unreachable" => counts.unreachable = n,
                _ => counts.failed = n,
            }
        }
        Ok(counts)
    }

    /// (error kind, pages) for failed and unreachable pages, by kind.
    pub fn failure_kinds(&self) -> Result<Vec<(String, usize)>> {
        let mut stmt = self.conn.prepare(
            "SELECT ifnull(error_kind, 'unknown'), COUNT(*) FROM pages
             WHERE status IN ('failed', 'unreachable') GROUP BY 1 ORDER BY 1",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
        rows.map(|row| {
            let (kind, n) = row?;
            Ok((kind, usize::try_from(n)?))
        })
        .collect()
    }

    /// Saves a stub for a page that couldn't be reached: a title and a
    /// one-line summary saying why, tagged `status/unreachable`.
    pub fn save_unreachable(
        &mut self,
        page_id: i64,
        stub: &PageResult,
        kind: &str,
        error: &str,
    ) -> Result<()> {
        self.save(page_id, "unreachable", Some((kind, error)), stub)
    }

    /// Saves a stub note for a page that loaded but couldn't be summarized,
    /// e.g. a PDF, tagged `status/failed`.
    pub fn save_failed(&mut self, page_id: i64, stub: &PageResult, kind: &str, error: &str) -> Result<()> {
        self.save(page_id, "failed", Some((kind, error)), stub)
    }

    pub fn save_result(&mut self, page_id: i64, result: &PageResult) -> Result<()> {
        self.save(page_id, "done", None, result)
    }

    fn save(
        &mut self,
        page_id: i64,
        status: &str,
        error: Option<(&str, &str)>,
        result: &PageResult,
    ) -> Result<()> {
        let (kind, message) = error.unzip();
        let tx = self.conn.transaction()?;
        tx.execute(
            "UPDATE pages SET status = ?2, error_kind = ?3, error = ?4, title = ?5, summary = ?6, lang = ?7,
                              processed_at = ?8
             WHERE id = ?1",
            params![
                page_id,
                status,
                kind,
                message,
                result.title,
                result.summary,
                result.lang,
                now()
            ],
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

    fn tag_id_where(&self, condition: &str, value: &str) -> Result<Option<i64>> {
        let sql = format!("SELECT id FROM tags WHERE {condition} LIMIT 1");
        Ok(self.conn.query_row(&sql, [value], |r| r.get(0)).optional()?)
    }

    /// Finds the existing tag a normalized name refers to: the tag itself, a
    /// recorded alias, or the same words without hyphens (`machinelearning` /
    /// `machine-learning`). Returns the tag id and whether it was found under a
    /// different name. Plural and singular forms are not matched here: `glasses`
    /// is not `glass`, so those merges are left to the reconciliation pass,
    /// where they are confirmed.
    pub fn find_tag(&self, name: &str) -> Result<Option<(i64, bool)>> {
        if let Some(id) = self.tag_id_where("name = ?1", name)? {
            return Ok(Some((id, false)));
        }
        let alias: Option<i64> = self
            .conn
            .query_row("SELECT tag_id FROM tag_aliases WHERE alias = ?1", [name], |r| {
                r.get(0)
            })
            .optional()?;
        if let Some(id) = alias {
            return Ok(Some((id, true)));
        }
        let squashed = name.replace('-', "");
        Ok(self
            .tag_id_where("replace(name, '-', '') = ?1", &squashed)?
            .map(|id| (id, true)))
    }

    pub fn create_tag(&self, name: &str, description: Option<&str>) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO tags (name, description, created_at) VALUES (?1, ?2, ?3)",
            params![name, description, now()],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Records that `alias` resolves to `tag_id`. An existing alias is kept.
    pub fn add_alias(&self, alias: &str, tag_id: i64, source: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO tag_aliases (alias, tag_id, source, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![alias, tag_id, source, now()],
        )?;
        Ok(())
    }

    pub fn tags(&self) -> Result<Vec<TagRow>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, parent_id, name, description FROM tags ORDER BY name")?;
        let rows = stmt.query_map([], |r| {
            Ok(TagRow {
                id: r.get(0)?,
                parent_id: r.get(1)?,
                name: r.get(2)?,
                description: r.get(3)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Distinct (page id, tag id) pairs for pages with notes: every page that
    /// isn't pending.
    pub fn tag_links(&self) -> Result<Vec<(i64, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT pt.page_id, pt.resolved_tag_id
             FROM page_tags pt JOIN pages p ON p.id = pt.page_id
             WHERE p.status != 'pending'",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Pages that get a note: done ones, and the unreachable and failed ones,
    /// which get stub notes.
    pub fn note_pages(&self) -> Result<Vec<DonePage>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, url, source, ifnull(title, url), ifnull(summary, ''), lang, note_file,
                    ifnull(processed_at, added_at)
             FROM pages WHERE status != 'pending' ORDER BY id",
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

    /// Every note file name in use, including those of pages that went back to
    /// pending with --retry-failed: they keep their file for when they're done.
    pub fn note_files(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT note_file FROM pages WHERE note_file IS NOT NULL")?;
        let rows = stmt.query_map([], |r| r.get(0))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn set_note_file(&self, page_id: i64, file: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE pages SET note_file = ?2 WHERE id = ?1",
            params![page_id, file],
        )?;
        Ok(())
    }

    pub fn put_embedding(&self, kind: EmbeddingKind, ref_id: i64, model: &str, vector: &[f32]) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (kind, ref_id, model, vector) VALUES (?1, ?2, ?3, ?4)",
            params![kind.as_str(), ref_id, model, vector_to_blob(vector)],
        )?;
        Ok(())
    }

    pub fn embeddings(&self, kind: EmbeddingKind, model: &str) -> Result<Vec<(i64, Vec<f32>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT ref_id, vector FROM embeddings WHERE kind = ?1 AND model = ?2")?;
        let rows = stmt.query_map(params![kind.as_str(), model], |r| {
            Ok((r.get::<_, i64>(0)?, blob_to_vector(&r.get::<_, Vec<u8>>(1)?)))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Text to embed for every tag that has no vector for `model` yet.
    pub fn tags_missing_embedding(&self, model: &str) -> Result<Vec<(i64, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT t.id, t.name, t.description FROM tags t
             WHERE NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.kind = 'tag' AND e.ref_id = t.id AND e.model = ?1)
             ORDER BY t.id",
        )?;
        let rows = stmt.query_map([model], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                tag_embedding_text(&r.get::<_, String>(1)?, r.get::<_, Option<String>>(2)?.as_deref()),
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Text to embed for every done page that has no vector for `model` yet.
    pub fn pages_missing_embedding(&self, model: &str) -> Result<Vec<(i64, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT p.id, p.title, p.summary FROM pages p
             WHERE p.status = 'done'
               AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.kind = 'page' AND e.ref_id = p.id AND e.model = ?1)
             ORDER BY p.id",
        )?;
        let rows = stmt.query_map([model], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                page_embedding_text(&r.get::<_, String>(1)?, &r.get::<_, String>(2)?),
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }
}

/// Version 3 adds the 'unreachable' page status and pages.error_kind. SQLite
/// can't change a CHECK constraint, so the pages table is rebuilt, keeping
/// every row and id. Pages that failed under version 2 have no stub note and
/// no cause (a 404 and a PDF were both just 'failed'), so they become pending
/// again: the next run processes them and records why they fail.
fn migrate_v2_to_v3(conn: &Connection) -> Result<()> {
    const COLUMNS: &str = "id, url, original_url, browser_title, source, status, error, title, summary, lang, \
                           note_file, added_at, processed_at";
    // Foreign keys can only be switched off outside a transaction.
    conn.pragma_update(None, "foreign_keys", "OFF")?;
    let result = conn.execute_batch(&format!(
        "BEGIN;
         {new_table}
         INSERT INTO pages_v3 ({COLUMNS}) SELECT {COLUMNS} FROM pages;
         UPDATE pages_v3 SET status = 'pending', error = NULL WHERE status = 'failed';
         DROP TABLE pages;
         ALTER TABLE pages_v3 RENAME TO pages;
         PRAGMA user_version = 3;
         COMMIT;",
        new_table = PAGES_TABLE.replace("CREATE TABLE pages (", "CREATE TABLE pages_v3 ("),
    ));
    if result.is_err() {
        conn.execute_batch("ROLLBACK").ok();
    }
    conn.pragma_update(None, "foreign_keys", "ON")?;
    result.context("upgrading the database to version 3")?;
    Ok(())
}

/// What is embedded for a tag: its name, which is kebab-case, plus its description.
pub fn tag_embedding_text(name: &str, description: Option<&str>) -> String {
    let words = name.replace('-', " ");
    match description {
        Some(d) if !d.trim().is_empty() => format!("{words}: {}", d.trim()),
        _ => words,
    }
}

/// What is embedded for a processed page: its title and summary.
pub fn page_embedding_text(title: &str, summary: &str) -> String {
    format!("{title}\n{summary}")
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
    fn find_tag_matches_variants_and_aliases() {
        let db = Db::open_in_memory().unwrap();
        let games = db.create_tag("board-games", Some("Board games")).unwrap();
        let ml = db.create_tag("machine-learning", None).unwrap();
        assert_eq!(db.find_tag("board-games").unwrap(), Some((games, false)));
        assert_eq!(db.find_tag("boardgames").unwrap(), Some((games, true)));
        // Plurals are not matched: glasses (eyewear) is not glass.
        assert_eq!(db.find_tag("board-game").unwrap(), None);
        assert_eq!(db.find_tag("machinelearning").unwrap(), Some((ml, true)));
        assert_eq!(db.find_tag("ml").unwrap(), None);
        db.add_alias("ml", ml, "user").unwrap();
        assert_eq!(db.find_tag("ml").unwrap(), Some((ml, true)));
        assert_eq!(db.find_tag("gardening").unwrap(), None);
    }

    #[test]
    fn save_result_replaces_tags() {
        let mut db = Db::open_in_memory().unwrap();
        db.add_page("https://a.com/", "https://a.com/", None, "import")
            .unwrap();
        let page = db.pending_pages().unwrap()[0].id;
        let a = db.create_tag("a", None).unwrap();
        let b = db.create_tag("b", None).unwrap();
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
                failed: 0,
                unreachable: 0
            }
        );
    }

    #[test]
    fn embeddings_round_trip_and_missing_lists() {
        let mut db = Db::open_in_memory().unwrap();
        let rust = db
            .create_tag("rust-programming", Some("The Rust language"))
            .unwrap();
        let go = db.create_tag("go", None).unwrap();
        db.put_embedding(EmbeddingKind::Tag, rust, "m", &[0.6, -0.8])
            .unwrap();
        assert_eq!(
            db.embeddings(EmbeddingKind::Tag, "m").unwrap(),
            [(rust, vec![0.6, -0.8])]
        );
        assert_eq!(db.tags_missing_embedding("m").unwrap(), [(go, "go".to_string())]);
        // A different model needs its own vectors.
        assert_eq!(
            db.tags_missing_embedding("other").unwrap(),
            [
                (rust, "rust programming: The Rust language".to_string()),
                (go, "go".to_string())
            ]
        );

        db.add_page("https://a.com/", "https://a.com/", None, "import")
            .unwrap();
        let page = db.pending_pages().unwrap()[0].id;
        assert!(
            db.pages_missing_embedding("m").unwrap().is_empty(),
            "pending pages are not embedded"
        );
        db.save_result(
            page,
            &PageResult {
                title: "T",
                summary: "S.",
                lang: None,
                tags: &[],
            },
        )
        .unwrap();
        assert_eq!(
            db.pages_missing_embedding("m").unwrap(),
            [(page, "T\nS.".to_string())]
        );
    }

    /// The version 2 schema, as released in phase 2a.
    const SCHEMA_V2_PAGES: &str = "
        CREATE TABLE pages (
            id INTEGER PRIMARY KEY, url TEXT NOT NULL UNIQUE, original_url TEXT NOT NULL, browser_title TEXT,
            source TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'done', 'failed')),
            error TEXT, title TEXT, summary TEXT, lang TEXT, note_file TEXT UNIQUE, added_at TEXT NOT NULL,
            processed_at TEXT
        );";

    #[test]
    fn migrates_version_two_keeping_pages_and_tags() {
        let file = tempfile::NamedTempFile::new().unwrap();
        {
            let conn = Connection::open(file.path()).unwrap();
            conn.execute_batch(SCHEMA_V2_PAGES).unwrap();
            conn.execute_batch(SCHEMA).unwrap();
            conn.execute_batch(
                "INSERT INTO pages (id, url, original_url, source, status, title, summary, note_file, added_at,
                                    processed_at)
                 VALUES (7, 'https://a.com/', 'https://a.com/', 'import', 'done', 'T', 'S.', 't.md', 'x', 'y'),
                        (8, 'https://b.com/', 'https://b.com/', 'import', 'failed', NULL, NULL, NULL, 'x', 'y');
                 INSERT INTO tags (id, name, created_at) VALUES (3, 'rust', 'x');
                 INSERT INTO page_tags VALUES (7, 'rust', 3);
                 PRAGMA user_version = 2;",
            )
            .unwrap();
        }
        let mut db = Db::open(file.path()).unwrap();
        let version: i32 = db
            .conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
        let pages = db.note_pages().unwrap();
        assert_eq!(
            (
                pages[0].id,
                pages[0].title.as_str(),
                pages[0].note_file.as_deref()
            ),
            (7, "T", Some("t.md"))
        );
        assert_eq!(db.tag_links().unwrap(), [(7, 3)]);
        let fk: bool = db
            .conn
            .pragma_query_value(None, "foreign_keys", |r| r.get(0))
            .unwrap();
        assert!(fk, "foreign keys are back on");

        // The page that failed under version 2 is pending again.
        assert_eq!(
            db.pending_pages()
                .unwrap()
                .iter()
                .map(|p| p.id)
                .collect::<Vec<_>>(),
            [8]
        );

        // The new status works.
        let stub = PageResult {
            title: "B",
            summary: "Unreachable.",
            lang: None,
            tags: &[],
        };
        db.save_unreachable(8, &stub, "not_found", "HTTP 404").unwrap();
        assert_eq!(
            db.status_counts().unwrap(),
            StatusCounts {
                pending: 0,
                done: 1,
                failed: 0,
                unreachable: 1
            }
        );
        assert_eq!(db.failure_kinds().unwrap(), [("not_found".to_string(), 1)]);
    }

    #[test]
    fn refuses_phase_one_databases() {
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "user_version", 1).unwrap();
        let err = Db::init(conn).err().unwrap();
        assert!(err.to_string().contains("new --out folder"), "{err}");
    }
}

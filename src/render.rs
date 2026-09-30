use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};

use crate::db::{Db, DonePage};
use crate::tags::{CountedTag, count_tree};

pub const NOTES_DIR: &str = "notes";
const MAX_SLUG_CHARS: usize = 80;

#[derive(Debug, PartialEq, Eq)]
pub struct RenderStats {
    pub notes: usize,
}

/// Writes one note per processed page plus `_tags.md` and `_index.md`.
/// Files in the notes folder that tabkeeper doesn't know about are left alone.
pub fn render_all(db: &Db, out_dir: &Path) -> Result<RenderStats> {
    let notes_dir = out_dir.join(NOTES_DIR);
    std::fs::create_dir_all(&notes_dir).with_context(|| format!("creating {}", notes_dir.display()))?;

    let mut pages = db.done_pages()?;
    assign_note_files(db, &mut pages)?;

    let tags = db.tags()?;
    let path_by_id: HashMap<i64, &str> = tags.iter().map(|t| (t.id, t.path.as_str())).collect();
    let mut tags_by_page: HashMap<i64, Vec<&str>> = HashMap::new();
    let links = db.tag_links()?;
    for (page_id, tag_id) in &links {
        if let Some(path) = path_by_id.get(tag_id) {
            tags_by_page.entry(*page_id).or_default().push(path);
        }
    }
    for list in tags_by_page.values_mut() {
        list.sort_unstable();
    }

    for page in &pages {
        let file = page.note_file.as_deref().expect("note files were just assigned");
        let page_tags = tags_by_page.get(&page.id).map(Vec::as_slice).unwrap_or_default();
        let path = notes_dir.join(file);
        std::fs::write(&path, note(page, page_tags))
            .with_context(|| format!("writing {}", path.display()))?;
    }

    let counted = count_tree(&tags, &links);
    std::fs::write(out_dir.join("_tags.md"), tags_markdown(&counted))?;
    std::fs::write(out_dir.join("_index.md"), index_markdown(&pages, &tags_by_page))?;
    Ok(RenderStats { notes: pages.len() })
}

/// Gives each page without a note file a unique name based on its title.
/// Names are stored so a note keeps its file name across renders.
fn assign_note_files(db: &Db, pages: &mut [DonePage]) -> Result<()> {
    let mut taken: HashSet<String> = pages.iter().filter_map(|p| p.note_file.clone()).collect();
    for page in pages.iter_mut().filter(|p| p.note_file.is_none()) {
        let base = slug(&page.title);
        let mut name = format!("{base}.md");
        let mut n = 2;
        while taken.contains(&name) {
            name = format!("{base}-{n}.md");
            n += 1;
        }
        db.set_note_file(page.id, &name)?;
        taken.insert(name.clone());
        page.note_file = Some(name);
    }
    Ok(())
}

pub fn slug(title: &str) -> String {
    let mut out = String::new();
    for c in title.chars().flat_map(char::to_lowercase) {
        if out.chars().count() >= MAX_SLUG_CHARS {
            break;
        }
        if c.is_alphanumeric() {
            out.push(c);
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    let trimmed = out.trim_end_matches('-');
    if trimmed.is_empty() {
        "page".to_string()
    } else {
        trimmed.to_string()
    }
}

/// A double-quoted YAML scalar. JSON string syntax is valid YAML.
fn yaml_str(s: &str) -> String {
    serde_json::to_string(s).expect("strings always serialize")
}

fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn note(page: &DonePage, tags: &[&str]) -> String {
    let mut out = String::from("---\n");
    writeln!(out, "url: {}", yaml_str(&page.url)).unwrap();
    writeln!(out, "title: {}", yaml_str(&one_line(&page.title))).unwrap();
    writeln!(out, "source: {}", page.source).unwrap();
    writeln!(out, "captured: {}", page.processed_at).unwrap();
    if let Some(lang) = &page.lang {
        writeln!(out, "lang: {}", yaml_str(lang)).unwrap();
    }
    if tags.is_empty() {
        out.push_str("tags: []\n");
    } else {
        out.push_str("tags:\n");
        for tag in tags {
            writeln!(out, "  - {tag}").unwrap();
        }
    }
    out.push_str("---\n\n");
    writeln!(out, "# {}\n", one_line(&page.title)).unwrap();
    writeln!(out, "**URL:** <{}>\n", page.url).unwrap();
    writeln!(out, "{}", page.summary.trim()).unwrap();
    if !tags.is_empty() {
        let hashtags: Vec<String> = tags.iter().map(|t| format!("#{t}")).collect();
        writeln!(out, "\n{}", hashtags.join(" ")).unwrap();
    }
    out
}

fn tags_markdown(counted: &[CountedTag]) -> String {
    let mut out = String::from(
        "# Tags\n\nPages per tag. The first number includes pages tagged with anything below the tag.\n\n",
    );
    for t in counted.iter().filter(|t| t.total > 0) {
        let indent = "  ".repeat(t.depth);
        write!(out, "{indent}- **{}**: {} pages", t.name, t.total).unwrap();
        if t.direct != t.total {
            write!(out, " ({} tagged directly)", t.direct).unwrap();
        }
        if let Some(d) = &t.description {
            write!(out, " - {}", one_line(d)).unwrap();
        }
        out.push('\n');
    }
    out
}

fn markdown_link_text(s: &str) -> String {
    one_line(s)
        .replace('\\', "\\\\")
        .replace('[', "\\[")
        .replace(']', "\\]")
}

fn index_markdown(pages: &[DonePage], tags_by_page: &HashMap<i64, Vec<&str>>) -> String {
    let mut groups: BTreeMap<&str, Vec<&DonePage>> = BTreeMap::new();
    let mut untagged = Vec::new();
    for page in pages {
        let roots: HashSet<&str> = tags_by_page
            .get(&page.id)
            .into_iter()
            .flatten()
            .map(|path| path.split('/').next().unwrap_or(path))
            .collect();
        if roots.is_empty() {
            untagged.push(page);
        }
        for root in roots {
            groups.entry(root).or_default().push(page);
        }
    }

    let mut out = format!("# Index\n\n{} pages.\n", pages.len());
    let mut section = |name: &str, list: &mut Vec<&DonePage>| {
        list.sort_by_key(|p| p.title.to_lowercase());
        write!(out, "\n## {name}\n\n").unwrap();
        for page in list.iter() {
            let file = page.note_file.as_deref().unwrap_or_default();
            writeln!(
                out,
                "- [{}]({NOTES_DIR}/{})",
                markdown_link_text(&page.title),
                file.replace(' ', "%20")
            )
            .unwrap();
        }
    };
    for (root, list) in groups.iter_mut() {
        section(root, list);
    }
    if !untagged.is_empty() {
        section("Untagged", &mut untagged);
    }
    out
}

/// The tag tree as a plain-text table for the terminal.
pub fn report(counted: &[CountedTag]) -> String {
    let rows: Vec<(String, usize, usize)> = counted
        .iter()
        .filter(|t| t.total > 0)
        .map(|t| (format!("{}{}", "  ".repeat(t.depth), t.name), t.total, t.direct))
        .collect();
    if rows.is_empty() {
        return "No tags yet.\n".into();
    }
    let width = rows.iter().map(|r| r.0.chars().count()).max().unwrap_or(0).max(3);
    let mut out = format!("{:<width$}  {:>6}  {:>6}\n", "TAG", "PAGES", "DIRECT");
    for (name, total, direct) in rows {
        writeln!(out, "{name:<width$}  {total:>6}  {direct:>6}").unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::PageResult;
    use crate::tags::TagPath;

    fn page(title: &str) -> DonePage {
        DonePage {
            id: 1,
            url: "https://example.com/a".into(),
            source: "import".into(),
            title: title.into(),
            summary: "Two sentences. About things.".into(),
            lang: Some("en".into()),
            note_file: None,
            processed_at: "2026-09-30T12:00:00Z".into(),
        }
    }

    #[test]
    fn slugs() {
        assert_eq!(
            slug("Fine-tuning small LLMs: a guide!"),
            "fine-tuning-small-llms-a-guide"
        );
        assert_eq!(slug("  ¿Qué es Rust?  "), "qué-es-rust");
        assert_eq!(slug("!!!"), "page");
        assert_eq!(slug(&"a".repeat(200)).chars().count(), MAX_SLUG_CHARS);
    }

    #[test]
    fn note_format() {
        let n = note(&page("Say \"hi\""), &["tech/ai/llm", "tech/hardware"]);
        assert_eq!(
            n,
            "---\nurl: \"https://example.com/a\"\ntitle: \"Say \\\"hi\\\"\"\nsource: import\ncaptured: 2026-09-30T12:00:00Z\nlang: \"en\"\ntags:\n  - tech/ai/llm\n  - tech/hardware\n---\n\n# Say \"hi\"\n\n**URL:** <https://example.com/a>\n\nTwo sentences. About things.\n\n#tech/ai/llm #tech/hardware\n"
        );
    }

    #[test]
    fn renders_files_with_stable_unique_names() {
        let mut db = Db::open_in_memory().unwrap();
        let tag = db
            .ensure_tag(&TagPath::parse("tech/rust", 3).unwrap(), None)
            .unwrap();
        for url in ["https://a.com/", "https://b.com/", "https://c.com/"] {
            db.add_page(url, url, None, "import").unwrap();
        }
        let ids: Vec<i64> = db.pending_pages().unwrap().iter().map(|p| p.id).collect();
        let tagged = [("tech/rust".to_string(), tag)];
        for (i, id) in ids.iter().enumerate() {
            let tags: &[(String, i64)] = if i < 2 { &tagged } else { &[] };
            db.save_result(
                *id,
                &PageResult {
                    title: "Same Title",
                    summary: "S.",
                    lang: None,
                    tags,
                },
            )
            .unwrap();
        }

        let dir = tempfile::tempdir().unwrap();
        assert_eq!(render_all(&db, dir.path()).unwrap(), RenderStats { notes: 3 });
        let files: Vec<Option<String>> = db
            .done_pages()
            .unwrap()
            .into_iter()
            .map(|p| p.note_file)
            .collect();
        assert_eq!(
            files,
            [
                Some("same-title.md".into()),
                Some("same-title-2.md".into()),
                Some("same-title-3.md".into())
            ]
        );
        assert!(dir.path().join("notes/same-title-2.md").exists());

        let index = std::fs::read_to_string(dir.path().join("_index.md")).unwrap();
        assert!(
            index.contains("## tech\n\n- [Same Title](notes/same-title.md)"),
            "{index}"
        );
        assert!(
            index.contains("## Untagged\n\n- [Same Title](notes/same-title-3.md)"),
            "{index}"
        );
        let tags_md = std::fs::read_to_string(dir.path().join("_tags.md")).unwrap();
        assert!(
            tags_md.contains("- **tech**: 2 pages (0 tagged directly)\n  - **rust**: 2 pages\n"),
            "{tags_md}"
        );

        // A second render keeps the same file names.
        render_all(&db, dir.path()).unwrap();
        let again: Vec<Option<String>> = db
            .done_pages()
            .unwrap()
            .into_iter()
            .map(|p| p.note_file)
            .collect();
        assert_eq!(files, again);
    }

    #[test]
    fn report_table() {
        let counted = vec![
            CountedTag {
                id: 1,
                path: "tech".into(),
                name: "tech".into(),
                depth: 0,
                direct: 0,
                total: 3,
                description: None,
            },
            CountedTag {
                id: 2,
                path: "tech/rust".into(),
                name: "rust".into(),
                depth: 1,
                direct: 3,
                total: 3,
                description: None,
            },
        ];
        assert_eq!(
            report(&counted),
            "TAG      PAGES  DIRECT\ntech         3       0\n  rust       3       3\n"
        );
    }
}

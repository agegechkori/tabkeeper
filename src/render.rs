use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};

use crate::config::TitleStyle;
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
pub fn render_all(db: &Db, out_dir: &Path, style: TitleStyle) -> Result<RenderStats> {
    let notes_dir = out_dir.join(NOTES_DIR);
    std::fs::create_dir_all(&notes_dir).with_context(|| format!("creating {}", notes_dir.display()))?;

    let mut pages = db.note_pages()?;
    assign_note_files(db, &mut pages, &notes_dir)?;

    let links = db.tag_links()?;
    let counted = count_tree(&db.tags()?, &links);
    let by_id: HashMap<i64, &CountedTag> = counted.iter().map(|t| (t.id, t)).collect();
    let mut tags_by_page: HashMap<i64, Vec<&CountedTag>> = HashMap::new();
    for (page_id, tag_id) in &links {
        if let Some(tag) = by_id.get(tag_id) {
            tags_by_page.entry(*page_id).or_default().push(tag);
        }
    }
    for list in tags_by_page.values_mut() {
        list.sort_unstable_by(|a, b| a.path.cmp(&b.path));
    }

    for page in &pages {
        let file = page.note_file.as_deref().expect("note files were just assigned");
        let page_tags: Vec<&str> = tags_by_page
            .get(&page.id)
            .into_iter()
            .flatten()
            .map(|t| t.path.as_str())
            .collect();
        let path = notes_dir.join(file);
        std::fs::write(&path, note(page, &page_tags, style))
            .with_context(|| format!("writing {}", path.display()))?;
    }

    std::fs::write(out_dir.join("_tags.md"), tags_markdown(&counted))?;
    std::fs::write(
        out_dir.join("_index.md"),
        index_markdown(&pages, &tags_by_page, style),
    )?;
    Ok(RenderStats { notes: pages.len() })
}

/// Gives each page without a note file a unique name based on its title.
/// Names are stored so a note keeps its file name across renders.
fn assign_note_files(db: &Db, pages: &mut [DonePage], notes_dir: &Path) -> Result<()> {
    // Every name in the database counts, not just these pages': a page sent
    // back to pending by --retry-failed keeps its file name. A file already
    // on disk that isn't ours (the user's own note, or one from a deleted
    // database) is never overwritten.
    let mut taken: HashSet<String> = db.note_files()?.into_iter().collect();
    for page in pages.iter_mut().filter(|p| p.note_file.is_none()) {
        // The page's own title is shorter and what you'd recognize.
        let base = slug(own_title(page).unwrap_or(&page.title));
        let mut name = format!("{base}.md");
        let mut n = 2;
        while taken.contains(&name) || notes_dir.join(&name).exists() {
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

/// Words only, lowercased, for comparing titles.
fn title_words(title: &str) -> Vec<String> {
    title
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Whether `needle`'s words appear in `haystack`, in order and next to each
/// other: "rust" is in "the rust book", not in "trust and safety".
fn contains_words(haystack: &[String], needle: &[String]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|window| window == needle)
}

/// The page's own title, if it has any letters or digits: a tab titled "—"
/// or with only an emoji says nothing about the page.
fn own_title(page: &DonePage) -> Option<&str> {
    page.page_title
        .as_deref()
        .map(str::trim)
        .filter(|t| t.chars().any(char::is_alphanumeric))
}

/// The page's own title, when it says something the model's title doesn't.
fn distinct_page_title(page: &DonePage) -> Option<&str> {
    let own = own_title(page)?;
    let (a, b) = (title_words(own), title_words(&page.title));
    (!contains_words(&a, &b) && !contains_words(&b, &a)).then_some(own)
}

/// The note's title for the configured style.
pub fn display_title(page: &DonePage, style: TitleStyle) -> String {
    let own = own_title(page);
    let title = match style {
        TitleStyle::Summary => page.title.clone(),
        TitleStyle::Page => own.unwrap_or(&page.title).to_string(),
        TitleStyle::Both => match (own, distinct_page_title(page)) {
            (_, Some(own)) => format!("{own} ({})", page.title.trim()),
            // The same words: the page's own title says it all.
            (Some(own), None) => own.to_string(),
            (None, None) => page.title.clone(),
        },
    };
    one_line(&title)
}

pub fn note(page: &DonePage, tags: &[&str], style: TitleStyle) -> String {
    let title = display_title(page, style);
    let mut out = String::from("---\n");
    writeln!(out, "url: {}", yaml_str(&page.url)).unwrap();
    writeln!(out, "title: {}", yaml_str(&title)).unwrap();
    if let Some(own) = distinct_page_title(page) {
        writeln!(out, "page_title: {}", yaml_str(&one_line(own))).unwrap();
        writeln!(out, "summary_title: {}", yaml_str(&one_line(&page.title))).unwrap();
    }
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
            // Quoted, so tags like `null` or `true` stay strings.
            writeln!(out, "  - {}", yaml_str(tag)).unwrap();
        }
    }
    out.push_str("---\n\n");
    writeln!(out, "# {title}\n").unwrap();
    writeln!(out, "**URL:** <{}>\n", page.url).unwrap();
    writeln!(out, "{}", page.summary.trim()).unwrap();
    if !tags.is_empty() {
        let hashtags: Vec<String> = tags.iter().map(|t| format!("#{t}")).collect();
        writeln!(out, "\n{}", hashtags.join(" ")).unwrap();
    }
    out
}

fn tags_markdown(counted: &[CountedTag]) -> String {
    let mut out = String::from("# Tags\n\nPages per tag, most used first.\n\n");
    for t in counted.iter().filter(|t| t.total > 0) {
        let indent = "  ".repeat(t.depth);
        let noun = if t.total == 1 { "page" } else { "pages" };
        write!(out, "{indent}- **{}**: {} {noun}", t.name, t.total).unwrap();
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

/// Lists every page once, under its most used tag, so sections are big
/// topics rather than one section per tag.
fn index_markdown(
    pages: &[DonePage],
    tags_by_page: &HashMap<i64, Vec<&CountedTag>>,
    style: TitleStyle,
) -> String {
    let mut groups: BTreeMap<&str, Vec<&DonePage>> = BTreeMap::new();
    let mut untagged = Vec::new();
    for page in pages {
        let main_tag = tags_by_page
            .get(&page.id)
            .into_iter()
            .flatten()
            .min_by(|a, b| b.total.cmp(&a.total).then(a.path.cmp(&b.path)));
        match main_tag {
            Some(tag) => groups.entry(tag.path.as_str()).or_default().push(page),
            None => untagged.push(page),
        }
    }

    let mut out = format!("# Index\n\n{} pages.\n", pages.len());
    let mut section = |name: &str, list: &mut Vec<&DonePage>| {
        list.sort_by_key(|p| display_title(p, style).to_lowercase());
        write!(out, "\n## {name}\n\n").unwrap();
        for page in list.iter() {
            let file = page.note_file.as_deref().unwrap_or_default();
            writeln!(
                out,
                "- [{}]({NOTES_DIR}/{})",
                markdown_link_text(&display_title(page, style)),
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
            page_title: None,
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
        let n = note(
            &page("Say \"hi\""),
            &["tech/ai/llm", "tech/hardware"],
            TitleStyle::Both,
        );
        assert_eq!(
            n,
            "---\nurl: \"https://example.com/a\"\ntitle: \"Say \\\"hi\\\"\"\nsource: import\ncaptured: 2026-09-30T12:00:00Z\nlang: \"en\"\ntags:\n  - \"tech/ai/llm\"\n  - \"tech/hardware\"\n---\n\n# Say \"hi\"\n\n**URL:** <https://example.com/a>\n\nTwo sentences. About things.\n\n#tech/ai/llm #tech/hardware\n"
        );
    }

    #[test]
    fn renders_files_with_stable_unique_names() {
        let mut db = Db::open_in_memory().unwrap();
        let rust = db
            .create_tag("rust-programming", Some("The Rust language"))
            .unwrap();
        let safety = db.create_tag("memory-safety", None).unwrap();
        for url in ["https://a.com/", "https://b.com/", "https://c.com/"] {
            db.add_page(url, url, None, "import").unwrap();
        }
        let ids: Vec<i64> = db.pending_pages().unwrap().iter().map(|p| p.id).collect();
        let both = [("Rust".to_string(), rust), ("memory-safety".to_string(), safety)];
        let one = [("rust-programming".to_string(), rust)];
        let tag_sets: [&[(String, i64)]; 3] = [&both, &one, &[]];
        for (id, tags) in ids.iter().zip(tag_sets) {
            db.save_result(
                *id,
                &PageResult {
                    title: "Same Title",
                    summary: "S.",
                    lang: None,
                    tags,
                    page_title: None,
                },
            )
            .unwrap();
        }

        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            render_all(&db, dir.path(), crate::config::TitleStyle::Both).unwrap(),
            RenderStats { notes: 3 }
        );
        let files: Vec<Option<String>> = db
            .note_pages()
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
        let first = std::fs::read_to_string(dir.path().join("notes/same-title.md")).unwrap();
        assert!(first.ends_with("#memory-safety #rust-programming\n"), "{first}");

        // Each page is listed once, under its most used tag.
        let index = std::fs::read_to_string(dir.path().join("_index.md")).unwrap();
        assert!(
            index.contains("## rust-programming\n\n- [Same Title](notes/same-title.md)\n- [Same Title](notes/same-title-2.md)\n"),
            "{index}"
        );
        assert!(!index.contains("## memory-safety"), "{index}");
        assert!(
            index.contains("## Untagged\n\n- [Same Title](notes/same-title-3.md)"),
            "{index}"
        );
        let tags_md = std::fs::read_to_string(dir.path().join("_tags.md")).unwrap();
        assert!(
            tags_md.contains(
                "- **rust-programming**: 2 pages - The Rust language\n- **memory-safety**: 1 page\n"
            ),
            "{tags_md}"
        );

        // A second render keeps the same file names.
        render_all(&db, dir.path(), crate::config::TitleStyle::Both).unwrap();
        let again: Vec<Option<String>> = db
            .note_pages()
            .unwrap()
            .into_iter()
            .map(|p| p.note_file)
            .collect();
        assert_eq!(files, again);
    }

    #[test]
    fn retried_page_keeps_its_file_name() {
        let mut db = Db::open_in_memory().unwrap();
        db.add_page("https://a.com/", "https://a.com/", Some("Same Title"), "import")
            .unwrap();
        let a = db.pending_pages().unwrap()[0].id;
        let stub = PageResult {
            title: "Same Title",
            summary: "Unreachable.",
            lang: None,
            tags: &[],
            page_title: None,
        };
        db.save_unreachable(a, &stub, "timeout", "timed out").unwrap();
        let dir = tempfile::tempdir().unwrap();
        render_all(&db, dir.path(), crate::config::TitleStyle::Both).unwrap();

        // Page a goes back to pending and the run stops before retrying it;
        // a new page with the same title must not take a's file name.
        db.retry_failed().unwrap();
        db.add_page("https://b.com/", "https://b.com/", None, "import")
            .unwrap();
        let b = db.pending_pages().unwrap().iter().find(|p| p.id != a).unwrap().id;
        db.save_result(
            b,
            &PageResult {
                title: "Same Title",
                summary: "S.",
                lang: None,
                tags: &[],
                page_title: None,
            },
        )
        .unwrap();
        render_all(&db, dir.path(), crate::config::TitleStyle::Both).unwrap();
        let files: Vec<_> = db
            .note_pages()
            .unwrap()
            .into_iter()
            .map(|p| (p.id, p.note_file))
            .collect();
        assert_eq!(files, [(b, Some("same-title-2.md".to_string()))]);
        assert_eq!(
            db.note_files().unwrap().len(),
            2,
            "a keeps same-title.md for when it's done"
        );
    }

    #[test]
    fn never_overwrites_files_it_did_not_write() {
        let mut db = Db::open_in_memory().unwrap();
        db.add_page("https://a.com/", "https://a.com/", None, "import")
            .unwrap();
        let id = db.pending_pages().unwrap()[0].id;
        db.save_result(
            id,
            &PageResult {
                title: "Rust ownership",
                summary: "S.",
                lang: None,
                tags: &[],
                page_title: None,
            },
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(NOTES_DIR)).unwrap();
        let mine = dir.path().join("notes/rust-ownership.md");
        std::fs::write(&mine, "my own note").unwrap();

        render_all(&db, dir.path(), crate::config::TitleStyle::Both).unwrap();
        assert_eq!(std::fs::read_to_string(&mine).unwrap(), "my own note");
        assert!(dir.path().join("notes/rust-ownership-2.md").exists());
    }

    #[test]
    fn tags_that_look_like_yaml_values_stay_strings() {
        let n = note(&page("T"), &["null", "true"], TitleStyle::Both);
        assert!(n.contains("tags:\n  - \"null\"\n  - \"true\"\n"), "{n}");
    }

    fn titled(own: Option<&str>, model: &str) -> DonePage {
        DonePage {
            page_title: own.map(str::to_string),
            ..page(model)
        }
    }

    #[test]
    fn title_styles() {
        let p = titled(Some("Kyoto - Wikipedia"), "Kyoto: Former Japanese Capital");
        assert_eq!(
            display_title(&p, TitleStyle::Both),
            "Kyoto - Wikipedia (Kyoto: Former Japanese Capital)"
        );
        assert_eq!(display_title(&p, TitleStyle::Page), "Kyoto - Wikipedia");
        assert_eq!(
            display_title(&p, TitleStyle::Summary),
            "Kyoto: Former Japanese Capital"
        );
        // The same words: only the page's own title.
        let same = titled(Some("Rust Programming Language"), "Rust programming language");
        assert_eq!(
            display_title(&same, TitleStyle::Both),
            "Rust Programming Language"
        );
        let contained = titled(Some("Sourdough"), "Sourdough bread");
        assert_eq!(display_title(&contained, TitleStyle::Both), "Sourdough");
        // Whole words only: "ai" is not in "detailed", "rust" not in "trust".
        let inside_a_word = titled(Some("AI"), "A detailed guide to tax filing");
        assert_eq!(
            display_title(&inside_a_word, TitleStyle::Both),
            "AI (A detailed guide to tax filing)"
        );
        let trust = titled(Some("Trust & Safety"), "Rust");
        assert_eq!(display_title(&trust, TitleStyle::Both), "Trust & Safety (Rust)");
        // A title without letters or digits is no title.
        for empty in ["—", "|", "...", "🙂"] {
            let p = titled(Some(empty), "The model's title");
            assert_eq!(
                display_title(&p, TitleStyle::Both),
                "The model's title",
                "{empty}"
            );
            assert_eq!(
                display_title(&p, TitleStyle::Page),
                "The model's title",
                "{empty}"
            );
        }
        // No page title (older notes): the model's title.
        let none = titled(None, "Model title");
        assert_eq!(display_title(&none, TitleStyle::Both), "Model title");
        assert_eq!(display_title(&none, TitleStyle::Page), "Model title");
    }

    #[test]
    fn front_matter_keeps_both_titles() {
        let n = note(
            &titled(Some("Kyoto - Wikipedia"), "Kyoto City"),
            &[],
            TitleStyle::Both,
        );
        assert!(
            n.contains("title: \"Kyoto - Wikipedia (Kyoto City)\"\npage_title: \"Kyoto - Wikipedia\"\nsummary_title: \"Kyoto City\"\n"),
            "{n}"
        );
        assert!(n.contains("\n# Kyoto - Wikipedia (Kyoto City)\n"), "{n}");
        let n = note(&titled(Some("Kyoto"), "Kyoto"), &[], TitleStyle::Both);
        assert!(!n.contains("page_title"), "nothing to add when they match: {n}");
    }

    #[test]
    fn new_note_files_are_named_after_the_page_title() {
        let mut db = Db::open_in_memory().unwrap();
        db.add_page("https://a.com/", "https://a.com/", None, "import")
            .unwrap();
        let id = db.pending_pages().unwrap()[0].id;
        db.save_result(
            id,
            &PageResult {
                title: "A long descriptive model title",
                summary: "S.",
                lang: None,
                tags: &[],
                page_title: None,
            },
        )
        .unwrap();
        db.set_page_title(id, "Short Tab Title").unwrap();
        let dir = tempfile::tempdir().unwrap();
        render_all(&db, dir.path(), TitleStyle::Both).unwrap();
        assert!(dir.path().join("notes/short-tab-title.md").exists());
        let index = std::fs::read_to_string(dir.path().join("_index.md")).unwrap();
        assert!(
            index.contains("[Short Tab Title (A long descriptive model title)](notes/short-tab-title.md)"),
            "{index}"
        );
    }

    #[test]
    fn file_name_ignores_a_page_title_without_words() {
        let mut db = Db::open_in_memory().unwrap();
        db.add_page("https://a.com/", "https://a.com/", None, "import")
            .unwrap();
        let id = db.pending_pages().unwrap()[0].id;
        db.save_result(
            id,
            &PageResult {
                title: "Model title",
                summary: "S.",
                lang: None,
                tags: &[],
                page_title: None,
            },
        )
        .unwrap();
        db.set_page_title(id, "—").unwrap();
        let dir = tempfile::tempdir().unwrap();
        render_all(&db, dir.path(), TitleStyle::Both).unwrap();
        assert!(dir.path().join("notes/model-title.md").exists());
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

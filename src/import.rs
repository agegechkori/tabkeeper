use crate::db::Db;
use crate::filter::Filter;
use crate::urls::{Rejected, normalize};

#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ImportStats {
    /// URLs in the list.
    pub listed: usize,
    pub added: usize,
    /// Repeats within the list, including the same page with different
    /// tracking parameters or fragments.
    pub duplicates: usize,
    /// Already in the archive from an earlier run or import.
    pub known: usize,
    pub filtered: usize,
    pub not_web: usize,
    pub invalid: Vec<String>,
}

/// Parses a URL list: one URL per line, optionally followed by whitespace and
/// a title. Blank lines and lines starting with `#` are ignored.
pub fn parse_url_list(text: &str) -> Vec<(&str, Option<&str>)> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|line| match line.split_once(char::is_whitespace) {
            Some((url, title)) => (url, Some(title.trim()).filter(|t| !t.is_empty())),
            None => (line, None),
        })
        .collect()
}

/// Adds the list's web URLs to the database, except those the filter excludes.
pub fn import_url_list(db: &Db, text: &str, filter: &Filter) -> anyhow::Result<ImportStats> {
    let mut stats = ImportStats::default();
    let mut seen = std::collections::HashSet::new();
    for (raw, title) in parse_url_list(text) {
        stats.listed += 1;
        match normalize(raw) {
            Ok(url) if !seen.insert(url.clone()) => stats.duplicates += 1,
            Ok(url) if !filter.allows(&url) => stats.filtered += 1,
            Ok(url) => {
                if db.add_page(&url, raw, title, "import")? {
                    stats.added += 1;
                } else {
                    stats.known += 1;
                }
            }
            Err(Rejected::NotWeb(_)) => stats.not_web += 1,
            Err(Rejected::Invalid) => stats.invalid.push(raw.to_string()),
        }
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lines() {
        let text = "# my tabs\nhttps://a.com/  A title here\n\n  https://b.com/\n";
        assert_eq!(
            parse_url_list(text),
            [("https://a.com/", Some("A title here")), ("https://b.com/", None)]
        );
    }

    #[test]
    fn imports_with_dedupe_and_filter() {
        let db = Db::open_in_memory().unwrap();
        let text = "https://a.com/?utm_source=x\nhttps://a.com/#frag\nchrome://newtab\nnonsense\nhttps://bank.example.com/\n";
        let config = crate::config::FilterConfig {
            mode: crate::config::FilterMode::Deny,
            rules: vec!["domain:bank.example.com".into()],
        };
        let filter = Filter::new(&config, &[], &[]).unwrap();
        let stats = import_url_list(&db, text, &filter).unwrap();
        assert_eq!(
            stats,
            ImportStats {
                listed: 5,
                added: 1,
                duplicates: 1,
                known: 0,
                filtered: 1,
                not_web: 1,
                invalid: vec!["nonsense".into()]
            }
        );
        assert_eq!(db.pending_pages().unwrap().len(), 1);
    }

    #[test]
    fn urls_already_in_the_archive_are_known() {
        let db = Db::open_in_memory().unwrap();
        let filter = Filter::default();
        import_url_list(&db, "https://a.com/\n", &filter).unwrap();
        let stats = import_url_list(&db, "https://a.com/#x\nhttps://b.com/\n", &filter).unwrap();
        assert_eq!(
            (stats.listed, stats.added, stats.known, stats.duplicates),
            (2, 1, 1, 0)
        );
    }
}

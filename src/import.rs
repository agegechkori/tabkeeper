use crate::db::Db;
use crate::urls::{Rejected, normalize};

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ImportStats {
    pub added: usize,
    pub known: usize,
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

pub fn import_url_list(db: &Db, text: &str) -> anyhow::Result<ImportStats> {
    let mut stats = ImportStats::default();
    for (raw, title) in parse_url_list(text) {
        match normalize(raw) {
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
    fn imports_with_dedupe() {
        let db = Db::open_in_memory().unwrap();
        let text = "https://a.com/?utm_source=x\nhttps://a.com/#frag\nchrome://newtab\nnonsense\n";
        let stats = import_url_list(&db, text).unwrap();
        assert_eq!(
            stats,
            ImportStats {
                added: 1,
                known: 1,
                not_web: 1,
                invalid: vec!["nonsense".into()]
            }
        );
    }
}

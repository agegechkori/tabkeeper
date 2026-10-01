use std::collections::{BTreeMap, HashMap, HashSet};

/// Prefix of tags the tool assigns itself, such as `status/unreachable`. A
/// model's tags are normalized to a single level, so they can never collide.
pub const RESERVED_PREFIX: &str = "status/";

/// Normalizes a tag as written by a model or a person to a flat,
/// lowercase kebab-case name. A path such as `technology/rust` keeps its most
/// specific level. Returns `None` for reserved or unusable tags.
pub fn normalize_name(raw: &str) -> Option<String> {
    let raw = raw.trim().trim_start_matches('#').trim();
    if raw.to_lowercase().starts_with(RESERVED_PREFIX) {
        return None;
    }
    let last = raw.split('/').rev().find_map(kebab_case)?;
    // Obsidian ignores tags made only of digits, like #2024.
    if last.chars().all(|c| c.is_ascii_digit() || c == '-') {
        return None;
    }
    Some(last)
}

fn kebab_case(raw: &str) -> Option<String> {
    let mut out = String::new();
    for c in raw.chars().flat_map(char::to_lowercase) {
        if c.is_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let trimmed = out.trim_matches('-');
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Singular and plural spellings of a tag's last word: `board-games` →
/// `board-game` and back. Only used to propose merges for the tag review to
/// confirm, since a plural can mean something else (`glasses`, `windows`).
pub fn plural_variants(name: &str) -> Vec<String> {
    let (head, last) = match name.rsplit_once('-') {
        Some((head, last)) => (Some(head), last),
        None => (None, name),
    };
    let mut words = Vec::new();
    if last.chars().count() > 3
        && last.ends_with('s')
        && !(last.ends_with("ss") || last.ends_with("us") || last.ends_with("is"))
    {
        if let Some(stem) = last.strip_suffix("ies") {
            words.push(format!("{stem}y"));
        }
        if let Some(stem) = last.strip_suffix("es") {
            words.push(stem.to_string());
        }
        words.push(last[..last.len() - 1].to_string());
    } else if !last.ends_with('s') {
        words.push(format!("{last}s"));
        if ["x", "z", "ch", "sh"].iter().any(|end| last.ends_with(end)) {
            words.push(format!("{last}es"));
        }
        if let Some(stem) = last.strip_suffix('y') {
            words.push(format!("{stem}ies"));
        }
    }
    words
        .into_iter()
        .filter(|w| w != last && !w.is_empty())
        .map(|w| match head {
            Some(head) => format!("{head}-{w}"),
            None => w,
        })
        .collect()
}

/// A tag as stored in the database.
#[derive(Debug, Clone)]
pub struct TagRow {
    pub id: i64,
    pub parent_id: Option<i64>,
    pub name: String,
    pub description: Option<String>,
}

/// A tag with page counts, in tree order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CountedTag {
    pub id: i64,
    /// The tag's name, or its path from the root once tags have parents.
    pub path: String,
    pub name: String,
    pub depth: usize,
    /// Pages tagged with exactly this tag.
    pub direct: usize,
    /// Distinct pages tagged with this tag or anything below it.
    pub total: usize,
    pub description: Option<String>,
}

/// Counts pages per tag and returns the tags depth-first, siblings ordered by
/// total pages (most first), then by name. Flat tags are all roots.
pub fn count_tree(tags: &[TagRow], links: &[(i64, i64)]) -> Vec<CountedTag> {
    let by_id: HashMap<i64, &TagRow> = tags.iter().map(|t| (t.id, t)).collect();
    let mut direct: HashMap<i64, HashSet<i64>> = HashMap::new();
    let mut subtree: HashMap<i64, HashSet<i64>> = HashMap::new();
    for &(page_id, tag_id) in links {
        direct.entry(tag_id).or_default().insert(page_id);
        let mut current = Some(tag_id);
        while let Some(id) = current {
            subtree.entry(id).or_default().insert(page_id);
            current = by_id.get(&id).and_then(|t| t.parent_id);
        }
    }

    let mut children: BTreeMap<Option<i64>, Vec<&TagRow>> = BTreeMap::new();
    for t in tags {
        children.entry(t.parent_id).or_default().push(t);
    }
    let len = |m: &HashMap<i64, HashSet<i64>>, id: i64| m.get(&id).map_or(0, HashSet::len);
    for list in children.values_mut() {
        list.sort_by(|a, b| {
            len(&subtree, b.id)
                .cmp(&len(&subtree, a.id))
                .then(a.name.cmp(&b.name))
        });
    }

    let mut out = Vec::with_capacity(tags.len());
    let mut stack: Vec<(&TagRow, usize, String)> = children
        .get(&None)
        .map(|roots| roots.iter().rev().map(|t| (*t, 0, t.name.clone())).collect())
        .unwrap_or_default();
    while let Some((tag, depth, path)) = stack.pop() {
        if let Some(kids) = children.get(&Some(tag.id)) {
            stack.extend(
                kids.iter()
                    .rev()
                    .map(|t| (*t, depth + 1, format!("{path}/{}", t.name))),
            );
        }
        out.push(CountedTag {
            id: tag.id,
            path,
            name: tag.name.clone(),
            depth,
            direct: len(&direct, tag.id),
            total: len(&subtree, tag.id),
            description: tag.description.clone(),
        });
    }
    out
}

/// One line per tag for the model: `name (pages) - description`, sorted by name.
pub fn format_vocabulary(tags: &[&CountedTag]) -> String {
    let mut sorted = tags.to_vec();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    sorted
        .iter()
        .map(|t| match &t.description {
            Some(d) => format!("{} ({}) - {}", t.name, t.total, d),
            None => format!("{} ({})", t.name, t.total),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Tags the model may reuse: used on at least one page, and not reserved.
pub fn is_offerable(tag: &CountedTag) -> bool {
    tag.total > 0 && !tag.name.starts_with(RESERVED_PREFIX)
}

/// The `limit` most used tags; the fallback when embeddings are unavailable.
pub fn most_used(counted: &[CountedTag], limit: usize) -> Vec<&CountedTag> {
    let mut top: Vec<&CountedTag> = counted.iter().filter(|t| is_offerable(t)).collect();
    top.sort_by(|a, b| b.total.cmp(&a.total).then(a.name.cmp(&b.name)));
    top.truncate(limit);
    top
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_to_flat_kebab_case() {
        assert_eq!(
            normalize_name("#Machine Learning").as_deref(),
            Some("machine-learning")
        );
        assert_eq!(normalize_name(" rust_lang ").as_deref(), Some("rust-lang"));
        assert_eq!(
            normalize_name("technology/Programming Languages/").as_deref(),
            Some("programming-languages")
        );
        assert_eq!(normalize_name("Café").as_deref(), Some("café"));
    }

    #[test]
    fn rejects_reserved_and_junk() {
        assert_eq!(normalize_name("status/unreachable"), None);
        assert_eq!(normalize_name("#Status/whatever"), None);
        assert_eq!(normalize_name("///"), None);
        assert_eq!(normalize_name("2024"), None);
        assert_eq!(normalize_name("web3").as_deref(), Some("web3"));
    }

    #[test]
    fn plural_variants_both_ways() {
        assert!(plural_variants("board-games").contains(&"board-game".to_string()));
        assert!(plural_variants("board-game").contains(&"board-games".to_string()));
        assert!(plural_variants("policies").contains(&"policy".to_string()));
        assert!(plural_variants("box").contains(&"boxes".to_string()));
        assert!(plural_variants("css").is_empty());
        assert!(plural_variants("analysis").is_empty());
    }

    fn row(id: i64, parent: Option<i64>, name: &str) -> TagRow {
        TagRow {
            id,
            parent_id: parent,
            name: name.into(),
            description: None,
        }
    }

    #[test]
    fn counts_subtrees_without_double_counting() {
        let tags = vec![
            row(1, None, "tech"),
            row(2, Some(1), "rust"),
            row(3, Some(1), "python"),
            row(4, None, "food"),
        ];
        // Page 10 is tagged with both rust and python: tech counts it once.
        let links = vec![(10, 2), (10, 3), (11, 2), (12, 4), (13, 1)];
        let counted = count_tree(&tags, &links);
        let summary: Vec<(&str, usize, usize, usize)> = counted
            .iter()
            .map(|t| (t.path.as_str(), t.depth, t.direct, t.total))
            .collect();
        assert_eq!(
            summary,
            [
                ("tech", 0, 1, 3),
                ("tech/rust", 1, 2, 2),
                ("tech/python", 1, 1, 1),
                ("food", 0, 1, 1)
            ]
        );
    }

    #[test]
    fn most_used_skips_unused_and_reserved() {
        let tags = vec![
            row(1, None, "b"),
            row(2, None, "a"),
            row(3, None, "c"),
            row(4, None, "status/unreachable"),
            row(5, None, "unused"),
        ];
        let links = vec![
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 3),
            (6, 3),
            (7, 4),
            (8, 4),
            (9, 4),
            (10, 4),
        ];
        let counted = count_tree(&tags, &links);
        let top = most_used(&counted, 2);
        assert_eq!(format_vocabulary(&top), "b (2)\nc (3)");
    }
}

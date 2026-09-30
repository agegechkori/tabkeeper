use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

/// Top-level segment reserved for tags the tool assigns itself, such as
/// `status/unreachable`. The model may not use it.
pub const RESERVED_ROOT: &str = "status";

/// A normalized hierarchical tag, e.g. `technology/programming-languages/rust`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TagPath(Vec<String>);

impl TagPath {
    /// Normalizes a tag as written by a model or a person: segments become
    /// lowercase kebab-case, and paths deeper than `max_depth` keep their
    /// most general levels. Returns `None` if nothing usable is left.
    pub fn parse(raw: &str, max_depth: usize) -> Option<Self> {
        let mut segments: Vec<String> = raw
            .trim()
            .trim_start_matches('#')
            .split('/')
            .filter_map(normalize_segment)
            .collect();
        segments.truncate(max_depth);
        if segments.is_empty() {
            return None;
        }
        // Obsidian ignores tags made only of digits, like #2024.
        if segments
            .iter()
            .all(|s| s.chars().all(|c| c.is_ascii_digit() || c == '-'))
        {
            return None;
        }
        Some(Self(segments))
    }

    pub fn segments(&self) -> &[String] {
        &self.0
    }

    pub fn root(&self) -> &str {
        &self.0[0]
    }

    pub fn is_reserved(&self) -> bool {
        self.root() == RESERVED_ROOT
    }

    /// Every path from the root down to and including this one.
    pub fn prefixes(&self) -> impl Iterator<Item = TagPath> + '_ {
        (1..=self.0.len()).map(|n| TagPath(self.0[..n].to_vec()))
    }
}

impl fmt::Display for TagPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.join("/"))
    }
}

fn normalize_segment(raw: &str) -> Option<String> {
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

/// A tag as stored in the database.
#[derive(Debug, Clone)]
pub struct TagRow {
    pub id: i64,
    pub parent_id: Option<i64>,
    pub name: String,
    pub path: String,
    pub description: Option<String>,
}

/// A tag with page counts, in tree order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CountedTag {
    pub id: i64,
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
/// total pages (most first), then by name.
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
    let mut stack: Vec<(&TagRow, usize)> = children
        .get(&None)
        .map(|roots| roots.iter().rev().map(|t| (*t, 0)).collect())
        .unwrap_or_default();
    while let Some((tag, depth)) = stack.pop() {
        out.push(CountedTag {
            id: tag.id,
            path: tag.path.clone(),
            name: tag.name.clone(),
            depth,
            direct: len(&direct, tag.id),
            total: len(&subtree, tag.id),
            description: tag.description.clone(),
        });
        if let Some(kids) = children.get(&Some(tag.id)) {
            stack.extend(kids.iter().rev().map(|t| (*t, depth + 1)));
        }
    }
    out
}

/// The tag vocabulary shown to the model: the `limit` most used tags,
/// listed alphabetically so related tags sit next to each other.
pub fn vocabulary(counted: &[CountedTag], limit: usize) -> String {
    let mut top: Vec<&CountedTag> = counted
        .iter()
        .filter(|t| t.total > 0 && t.path.split('/').next() != Some(RESERVED_ROOT))
        .collect();
    top.sort_by(|a, b| b.total.cmp(&a.total).then(a.path.cmp(&b.path)));
    top.truncate(limit);
    top.sort_by(|a, b| a.path.cmp(&b.path));
    top.iter()
        .map(|t| match &t.description {
            Some(d) => format!("{} ({}) - {}", t.path, t.total, d),
            None => format!("{} ({})", t.path, t.total),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(raw: &str) -> Option<String> {
        TagPath::parse(raw, 3).map(|t| t.to_string())
    }

    #[test]
    fn normalizes_segments() {
        assert_eq!(
            p("#Technology/Programming Languages/Rust").as_deref(),
            Some("technology/programming-languages/rust")
        );
        assert_eq!(p(" machine_learning ").as_deref(), Some("machine-learning"));
        assert_eq!(p("science//chemistry/").as_deref(), Some("science/chemistry"));
        assert_eq!(p("food/Café").as_deref(), Some("food/café"));
    }

    #[test]
    fn truncates_depth_and_rejects_junk() {
        assert_eq!(p("a/b/c/d/e").as_deref(), Some("a/b/c"));
        assert_eq!(p("///"), None);
        assert_eq!(p("2024"), None);
        assert_eq!(p("news/2024").as_deref(), Some("news/2024"));
    }

    #[test]
    fn reserved_and_prefixes() {
        let t = TagPath::parse("status/unreachable", 3).unwrap();
        assert!(t.is_reserved());
        let t = TagPath::parse("a/b/c", 3).unwrap();
        let prefixes: Vec<String> = t.prefixes().map(|p| p.to_string()).collect();
        assert_eq!(prefixes, ["a", "a/b", "a/b/c"]);
    }

    fn row(id: i64, parent: Option<i64>, path: &str) -> TagRow {
        TagRow {
            id,
            parent_id: parent,
            name: path.rsplit('/').next().unwrap().into(),
            path: path.into(),
            description: None,
        }
    }

    #[test]
    fn counts_subtrees_without_double_counting() {
        let tags = vec![
            row(1, None, "tech"),
            row(2, Some(1), "tech/rust"),
            row(3, Some(1), "tech/python"),
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
                ("food", 0, 1, 1),
            ]
        );
    }

    #[test]
    fn vocabulary_keeps_most_used_sorted_by_path() {
        let tags = vec![row(1, None, "b"), row(2, None, "a"), row(3, None, "c")];
        let links = vec![(1, 1), (2, 1), (3, 2), (4, 3), (5, 3), (6, 3)];
        let vocab = vocabulary(&count_tree(&tags, &links), 2);
        assert_eq!(vocab, "b (2)\nc (3)");
    }
}

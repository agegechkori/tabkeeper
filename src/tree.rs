//! The tag tree: putting tags that aren't in it yet under categories, as
//! part of the tag review in hierarchical mode.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::json;

use crate::config::ReconcileConfig;
use crate::db::{Db, TagChange, TagInfo};
use crate::embed::{BATCH_SIZE, Embedder, similarity};
use crate::llm::Reviewer;
use crate::reconcile::{Proposed, describe_tag};
use crate::tags::normalize_name;

/// Categories shown to the model; more are left out.
const MAX_CATEGORIES_SHOWN: usize = 400;
/// Tag names listed in a proposal's description; more are counted.
const NAMES_SHOWN: usize = 8;

/// Your own top-level categories, from a TOML file.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Taxonomy {
    /// Only these categories may be at the top. Otherwise they are a starting
    /// point, and the model may add others.
    #[serde(default)]
    pub strict: bool,
    pub categories: Vec<Category>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Category {
    Name(String),
    Described {
        name: String,
        description: Option<String>,
    },
}

impl Category {
    fn name(&self) -> &str {
        match self {
            Category::Name(name) | Category::Described { name, .. } => name,
        }
    }

    fn description(&self) -> Option<&str> {
        match self {
            Category::Name(_) => None,
            Category::Described { description, .. } => description.as_deref(),
        }
    }
}

impl Taxonomy {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut taxonomy: Taxonomy =
            toml::from_str(&text).with_context(|| format!("reading {}", path.display()))?;
        for category in &mut taxonomy.categories {
            let name = normalize_name(category.name()).with_context(|| {
                format!(
                    "{}: {:?} isn't a usable category name",
                    path.display(),
                    category.name()
                )
            })?;
            match category {
                Category::Name(n) | Category::Described { name: n, .. } => *n = name,
            }
        }
        if taxonomy.categories.is_empty() {
            anyhow::bail!("{} lists no categories", path.display());
        }
        Ok(taxonomy)
    }

    fn names(&self) -> HashSet<&str> {
        self.categories.iter().map(Category::name).collect()
    }
}

const DOMAIN_SYSTEM: &str = "You sort the tags of a personal archive of web pages into broad domains, the top level of a tag tree. For each numbered tag, choose the one domain it belongs to, by what it means on its pages, not only by its name: rust-corrosion is science; rust used for the programming language is technology.

- Domains are broad, such as technology, science, business, finance, health, food, travel, entertainment, sports, politics, history, culture, education, home or shopping.
- Use an existing domain whenever one fits; add a new one only for a truly different domain.
- A tag that is itself a broad domain, like history or music, is its own domain.{taxonomy}

Reply with a JSON object: {\"domains\": [{\"tag\": 1, \"domain\": \"technology\"}, ...]}, one per tag, each domain a lowercase kebab-case English name.";

const BRANCH_SYSTEM: &str = "You organize one domain of the tag tree of a personal archive of web pages: {domain}. For each numbered tag, give path: the categories between {domain} and the tag, at most {levels}, each a lowercase kebab-case English name.

- Group related tags under a shared category, e.g. rust and python under programming-languages. Reuse existing categories wherever they fit.
- Create a category only when it groups several of these tags. A tag that fits no group gets an empty path and goes directly under {domain}.
- A broad tag can be a category itself: put narrower tags under it by naming it in their path.

Reply with a JSON object: {\"placements\": [{\"tag\": 1, \"path\": [\"programming-languages\"]}, ...]}, one per tag.";

/// Tags per request when sorting them into domains, which small models do
/// best a few at a time.
const DOMAIN_BATCH: usize = 10;
/// Tags per request when organizing a domain: enough to see what belongs
/// together.
const BRANCH_BATCH: usize = 40;

#[derive(Deserialize)]
struct DomainReply {
    domains: Vec<DomainDecision>,
}

#[derive(Deserialize)]
struct DomainDecision {
    tag: usize,
    domain: String,
}

#[derive(Deserialize)]
struct BranchReply {
    placements: Vec<BranchDecision>,
}

#[derive(Deserialize)]
struct BranchDecision {
    tag: usize,
    #[serde(default)]
    path: Vec<String>,
}

/// What the placement step proposes.
#[derive(Debug, Default)]
pub struct Placement {
    pub proposed: Vec<Proposed>,
    /// Tags that weren't in the tree, and how many of them got a proposal.
    pub unplaced: usize,
    pub placed: usize,
    pub failed_requests: Vec<String>,
}

/// Where tags are and will be in the tree, by name: the categories above
/// each one. Proposals from earlier batches count, so later batches build on
/// them, and a placement that contradicts them is turned down.
struct TreeModel {
    parents: HashMap<String, Vec<String>>,
    /// Names of every tag and alias, to the tag's name.
    canonical: HashMap<String, String>,
}

impl TreeModel {
    fn new(db: &Db) -> Result<Self> {
        let rows = db.tags()?;
        let by_id: HashMap<i64, &crate::tags::TagRow> = rows.iter().map(|t| (t.id, t)).collect();
        let mut parents = HashMap::new();
        for tag in rows.iter().filter(|t| t.placed) {
            let mut path = Vec::new();
            let mut current = tag.parent_id;
            while let Some(id) = current {
                let Some(parent) = by_id.get(&id) else { break };
                if path.len() > 64 {
                    break;
                }
                path.push(parent.name.clone());
                current = parent.parent_id;
            }
            path.reverse();
            parents.insert(tag.name.clone(), path);
        }
        let mut canonical: HashMap<String, String> =
            rows.iter().map(|t| (t.name.clone(), t.name.clone())).collect();
        for (alias, tag_id) in db.aliases()? {
            if let Some(tag) = by_id.get(&tag_id) {
                canonical.entry(alias).or_insert_with(|| tag.name.clone());
            }
        }
        Ok(Self { parents, canonical })
    }

    /// Category paths, for the model: every top-level tag and every tag
    /// something is under.
    fn categories(&self) -> Vec<String> {
        let mut names: HashSet<&str> = self
            .parents
            .values()
            .flat_map(|path| path.iter().map(String::as_str))
            .collect();
        names.extend(
            self.parents
                .iter()
                .filter(|(_, p)| p.is_empty())
                .map(|(n, _)| n.as_str()),
        );
        let mut out: Vec<String> = names
            .into_iter()
            .map(|name| {
                let mut path = self.parents.get(name).cloned().unwrap_or_default();
                path.push(name.to_string());
                path.join("/")
            })
            .collect();
        out.sort();
        out
    }

    /// The parent the tag gets, or `None` if the placement can't be used. A
    /// path that runs into a category somewhere else in the tree (names are
    /// unique, so a category is in one place), the tag itself or a repeated
    /// name is cut short there: the tag stays in the part that fits.
    fn check(
        &self,
        tag: &str,
        raw: &[String],
        levels: usize,
        strict: Option<&HashSet<&str>>,
    ) -> Option<Vec<String>> {
        // A tag already used as a category by an earlier placement stays there.
        if let Some(parent) = self.parents.get(tag) {
            return Some(parent.clone());
        }
        let mut parent: Vec<String> = Vec::new();
        for segment in raw.iter().take(levels) {
            let name = normalize_name(segment)?;
            let name = self.canonical.get(&name).cloned().unwrap_or(name);
            let elsewhere = self.parents.get(&name).is_some_and(|p| p[..] != parent[..]);
            if name == tag || parent.contains(&name) || elsewhere {
                break;
            }
            parent.push(name);
        }
        if let Some(allowed) = strict {
            let top = parent.first().map_or(tag, String::as_str);
            if !allowed.contains(top) {
                return None;
            }
        }
        Some(parent)
    }

    fn commit(&mut self, tag: &str, parent: &[String]) {
        for (i, name) in parent.iter().enumerate() {
            self.parents
                .entry(name.clone())
                .or_insert_with(|| parent[..i].to_vec());
        }
        self.parents.insert(tag.to_string(), parent.to_vec());
    }
}

fn place_key(tag: &str, parent: &[String]) -> String {
    format!("place:{tag}>{}", parent.join("/"))
}

/// Proposes a place in the tree for every topic tag that isn't in it yet.
pub async fn place<R: Reviewer, E: Embedder>(
    db: &Db,
    reviewer: &R,
    embedder: Option<&E>,
    config: &ReconcileConfig,
    max_depth: usize,
    taxonomy: Option<&Taxonomy>,
    note: impl Fn(&str),
) -> Result<Placement> {
    let mut placement = Placement::default();
    let levels = max_depth.saturating_sub(1);
    let mut unplaced: Vec<TagInfo> = db
        .topic_tags()?
        .into_iter()
        .filter(|t| !t.placed && !t.locked)
        .collect();
    placement.unplaced = unplaced.len();
    if unplaced.is_empty() || levels == 0 {
        return Ok(placement);
    }
    if let Some(embedder) = embedder {
        unplaced = similar_together(unplaced, embedder).await?;
    }
    note(&format!("Placing {} tags in the tag tree.", unplaced.len()));

    let mut model = TreeModel::new(db)?;
    let strict = taxonomy.filter(|t| t.strict).map(Taxonomy::names);

    // First each tag's domain, then, a domain at a time, where it goes in it:
    // seeing a domain's tags together, a model groups them consistently.
    let mut by_domain: BTreeMap<String, Vec<&TagInfo>> = BTreeMap::new();
    let domain_system = DOMAIN_SYSTEM.replace("{taxonomy}", &taxonomy_rules(taxonomy));
    let domain_schema = json!({
        "type": "object", "additionalProperties": false, "required": ["domains"],
        "properties": {"domains": {"type": "array", "items": {
            "type": "object", "additionalProperties": false, "required": ["tag", "domain"],
            "properties": {"tag": {"type": "integer"}, "domain": {"type": "string"}}
        }}}
    });
    for batch in unplaced.chunks(DOMAIN_BATCH.min(config.batch_size)) {
        let domains: Vec<String> = model
            .categories()
            .into_iter()
            .filter(|c| !c.contains('/'))
            .collect();
        let mut user = format!(
            "Existing domains: {}\n\nTags:\n",
            if domains.is_empty() {
                "(none yet)".to_string()
            } else {
                domains.join(", ")
            }
        );
        for (i, tag) in batch.iter().enumerate() {
            user.push_str(&format!("{}. {}\n", i + 1, describe_tag(db, tag)?));
        }
        let reply =
            match ask::<DomainReply, _>(reviewer, &domain_system, &user, "tag_domains", &domain_schema).await
            {
                Ok(reply) => reply,
                Err(err) => {
                    placement.failed_requests.push(err);
                    continue;
                }
            };
        for decision in reply.domains {
            let Some(tag) = decision.tag.checked_sub(1).and_then(|i| batch.get(i)) else {
                continue;
            };
            let Some(domain) = normalize_name(&decision.domain) else {
                continue;
            };
            let domain = model.canonical.get(&domain).cloned().unwrap_or(domain);
            if strict
                .as_ref()
                .is_some_and(|allowed| !allowed.contains(domain.as_str()))
            {
                continue;
            }
            let list = by_domain.entry(domain).or_default();
            if !list.iter().any(|t| t.id == tag.id) {
                list.push(tag);
            }
        }
    }

    let branch_schema = json!({
        "type": "object", "additionalProperties": false, "required": ["placements"],
        "properties": {"placements": {"type": "array", "items": {
            "type": "object", "additionalProperties": false, "required": ["tag", "path"],
            "properties": {"tag": {"type": "integer"}, "path": {"type": "array", "items": {"type": "string"}}}
        }}}
    });
    let mut chosen: Vec<(&TagInfo, Vec<String>)> = Vec::new();
    for (domain, tags) in &by_domain {
        // A tag that is its own domain goes at the top.
        let (own, rest): (Vec<&TagInfo>, Vec<&TagInfo>) = tags.iter().partition(|t| &t.name == domain);
        for tag in own {
            choose(db, &mut model, &mut chosen, tag, &[], levels, strict.as_ref())?;
        }
        if levels == 1 {
            for tag in rest {
                choose(
                    db,
                    &mut model,
                    &mut chosen,
                    tag,
                    std::slice::from_ref(domain),
                    levels,
                    strict.as_ref(),
                )?;
            }
            continue;
        }
        let system = BRANCH_SYSTEM
            .replace("{domain}", domain)
            .replace("{levels}", &(levels - 1).to_string());
        for batch in rest.chunks(BRANCH_BATCH) {
            let prefix = format!("{domain}/");
            let mut categories: Vec<String> = model
                .categories()
                .into_iter()
                .filter_map(|c| c.strip_prefix(&prefix).map(str::to_string))
                .collect();
            let more = categories.len().saturating_sub(MAX_CATEGORIES_SHOWN);
            categories.truncate(MAX_CATEGORIES_SHOWN);
            let mut user = format!("Existing categories in {domain}:\n");
            if categories.is_empty() {
                user.push_str("(none yet)\n");
            }
            for path in &categories {
                user.push_str(path);
                user.push('\n');
            }
            if more > 0 {
                user.push_str(&format!("(and {more} more)\n"));
            }
            user.push_str("\nTags:\n");
            for (i, tag) in batch.iter().enumerate() {
                user.push_str(&format!("{}. {}\n", i + 1, describe_tag(db, tag)?));
            }
            let reply =
                match ask::<BranchReply, _>(reviewer, &system, &user, "tag_branch", &branch_schema).await {
                    Ok(reply) => reply,
                    Err(err) => {
                        placement.failed_requests.push(err);
                        continue;
                    }
                };
            for decision in reply.placements {
                let Some(tag) = decision.tag.checked_sub(1).and_then(|i| batch.get(i)) else {
                    continue;
                };
                let mut parent = vec![domain.clone()];
                parent.extend(decision.path);
                choose(db, &mut model, &mut chosen, tag, &parent, levels, strict.as_ref())?;
            }
        }
    }
    placement.placed = chosen.len();
    placement.proposed = group(db, chosen)?;
    Ok(placement)
}

/// Asks the model and reads its reply; an error is described for the report.
async fn ask<T: serde::de::DeserializeOwned, R: Reviewer>(
    reviewer: &R,
    system: &str,
    user: &str,
    name: &str,
    schema: &serde_json::Value,
) -> std::result::Result<T, String> {
    let reply = reviewer
        .ask_json(system, user, name, schema.clone())
        .await
        .map_err(|err| format!("{err:#}"))?;
    serde_json::from_value(reply).map_err(|err| format!("the model's reply was unusable: {err}"))
}

/// Takes a placement if it fits the tree and wasn't declined before.
fn choose<'t>(
    db: &Db,
    model: &mut TreeModel,
    chosen: &mut Vec<(&'t TagInfo, Vec<String>)>,
    tag: &'t TagInfo,
    raw: &[String],
    levels: usize,
    strict: Option<&HashSet<&str>>,
) -> Result<()> {
    if chosen.iter().any(|(t, _)| t.id == tag.id) {
        return Ok(());
    }
    let Some(parent) = model.check(&tag.name, raw, levels, strict) else {
        return Ok(());
    };
    if db.tag_decision(&place_key(&tag.name, &parent))?.is_some() {
        return Ok(());
    }
    model.commit(&tag.name, &parent);
    chosen.push((tag, parent));
    Ok(())
}

fn taxonomy_rules(taxonomy: Option<&Taxonomy>) -> String {
    let Some(taxonomy) = taxonomy else {
        return String::new();
    };
    let mut out = if taxonomy.strict {
        String::from("\n- The domain must be one of these:\n")
    } else {
        String::from("\n- Prefer these domains, adding others only when none fits:\n")
    };
    for category in &taxonomy.categories {
        match category.description() {
            Some(d) => out.push_str(&format!("  - {}: {d}\n", category.name())),
            None => out.push_str(&format!("  - {}\n", category.name())),
        }
    }
    out.trim_end().to_string()
}

/// One proposal per parent, parents first, so a first build of hundreds of
/// tags is a list you can read.
fn group(db: &Db, chosen: Vec<(&TagInfo, Vec<String>)>) -> Result<Vec<Proposed>> {
    let mut groups: BTreeMap<(usize, Vec<String>), Vec<&TagInfo>> = BTreeMap::new();
    for (tag, parent) in chosen {
        groups.entry((parent.len(), parent)).or_default().push(tag);
    }
    let mut created: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for ((_, parent), mut tags) in groups {
        tags.sort_by(|a, b| b.pages.cmp(&a.pages).then(a.name.cmp(&b.name)));
        let mut new = Vec::new();
        for name in &parent {
            if db.tag_named(name)?.is_none() && created.insert(name.clone()) {
                new.push(name.as_str());
            }
        }
        let names: Vec<&str> = tags.iter().take(NAMES_SHOWN).map(|t| t.name.as_str()).collect();
        let mut description = format!(
            "place  {} ← {}",
            if parent.is_empty() {
                "(top level)".to_string()
            } else {
                parent.join("/")
            },
            names.join(", ")
        );
        if tags.len() > NAMES_SHOWN {
            description.push_str(&format!(" and {} more", tags.len() - NAMES_SHOWN));
        }
        if !new.is_empty() {
            description.push_str(&format!(" (new: {})", new.join(", ")));
        }
        out.push(Proposed {
            decision_keys: tags.iter().map(|t| place_key(&t.name, &parent)).collect(),
            change: TagChange::Place {
                parent,
                tags: tags.iter().map(|t| t.id).collect(),
            },
            description,
        });
    }
    Ok(out)
}

/// Orders tags so similar ones are next to each other and land in the same
/// batch, where the model places them consistently.
async fn similar_together<E: Embedder>(tags: Vec<TagInfo>, embedder: &E) -> Result<Vec<TagInfo>> {
    // The ordering is quadratic; past this, batches stay in usage order.
    const MAX: usize = 3000;
    if tags.len() < 3 || tags.len() > MAX {
        return Ok(tags);
    }
    let names: Vec<String> = tags.iter().map(|t| t.name.replace('-', " ")).collect();
    let mut vectors = Vec::with_capacity(names.len());
    for chunk in names.chunks(BATCH_SIZE) {
        vectors.extend(embedder.embed(chunk).await?);
    }
    let order = nearest_neighbour_order(&vectors);
    let mut slots: Vec<Option<TagInfo>> = tags.into_iter().map(Some).collect();
    Ok(order.into_iter().filter_map(|i| slots[i].take()).collect())
}

/// Starts at the first vector and keeps going to the most similar one not
/// visited yet.
fn nearest_neighbour_order(vectors: &[Vec<f32>]) -> Vec<usize> {
    let mut order = Vec::with_capacity(vectors.len());
    let mut visited = vec![false; vectors.len()];
    let mut current = 0;
    for _ in 0..vectors.len() {
        visited[current] = true;
        order.push(current);
        let next = (0..vectors.len()).filter(|&i| !visited[i]).max_by(|&a, &b| {
            similarity(&vectors[current], &vectors[a]).total_cmp(&similarity(&vectors[current], &vectors[b]))
        });
        match next {
            Some(next) => current = next,
            None => break,
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taxonomy_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("taxonomy.toml");
        std::fs::write(
            &path,
            "strict = true\ncategories = [\"Technology\", { name = \"food\", description = \"cooking\" }]\n",
        )
        .unwrap();
        let t = Taxonomy::load(&path).unwrap();
        assert!(t.strict);
        assert_eq!(t.names(), HashSet::from(["technology", "food"]));
        assert!(taxonomy_rules(Some(&t)).contains("  - food: cooking"));
        std::fs::write(&path, "categories = []\n").unwrap();
        assert!(Taxonomy::load(&path).is_err());
        let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("taxonomy.example.toml");
        let t = Taxonomy::load(&example).unwrap();
        assert!(!t.strict && t.names().contains("technology"));
    }

    fn model(placed: &[(&str, &[&str])]) -> TreeModel {
        TreeModel {
            parents: placed
                .iter()
                .map(|(n, p)| (n.to_string(), p.iter().map(|s| s.to_string()).collect()))
                .collect(),
            canonical: HashMap::from([("golang".to_string(), "go".to_string())]),
        }
    }

    fn path(p: &[&str]) -> Vec<String> {
        p.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn placements_are_checked_against_the_tree() {
        let mut m = model(&[("technology", &[]), ("programming-languages", &["technology"])]);
        // Normalized and trimmed to the depth.
        assert_eq!(
            m.check(
                "rust",
                &path(&["Technology", "Programming Languages", "systems"]),
                2,
                None
            ),
            Some(path(&["technology", "programming-languages"]))
        );
        // A category somewhere else in the tree cuts the path short.
        assert_eq!(
            m.check("rust", &path(&["science", "programming-languages"]), 2, None),
            Some(path(&["science"]))
        );
        // So does the tag itself.
        assert_eq!(
            m.check("rust", &path(&["technology", "rust"]), 2, None),
            Some(path(&["technology"]))
        );
        // Aliases mean their tag.
        assert_eq!(
            m.check("gin", &path(&["technology", "golang"]), 2, None),
            Some(path(&["technology", "go"]))
        );
        // A strict taxonomy limits the top level.
        let allowed = HashSet::from(["technology"]);
        assert_eq!(m.check("rust", &path(&["science"]), 2, Some(&allowed)), None);
        assert_eq!(m.check("science", &[], 2, Some(&allowed)), None);

        // Later placements build on earlier ones.
        m.commit("go", &path(&["technology", "programming-languages"]));
        assert_eq!(
            m.check("go", &path(&["games"]), 2, None),
            Some(path(&["technology", "programming-languages"]))
        );
        m.commit("chemistry", &path(&["science"]));
        assert_eq!(
            m.check("rust-corrosion", &path(&["technology", "chemistry"]), 2, None),
            Some(path(&["technology"]))
        );
        // Categories are what other tags are under, and the top level.
        let categories = m.categories();
        assert!(categories.contains(&"science".to_string()), "{categories:?}");
        assert!(
            !categories.contains(&"science/chemistry".to_string()),
            "a leaf so far"
        );
        m.commit("oxidation", &path(&["science", "chemistry"]));
        assert!(m.categories().contains(&"science/chemistry".to_string()));
    }

    #[tokio::test]
    async fn proposes_one_change_per_parent() {
        use crate::reconcile::tests::{FakeReviewer, archive, no_names, tag_id};
        let db = archive(&[
            (&["rust"], [1.0, 0.0, 0.0]),
            (&["python"], [1.0, 0.1, 0.0]),
            (&["rust-corrosion"], [0.0, 0.0, 1.0]),
            (&["history"], [0.0, 1.0, 0.0]),
        ]);
        // Numbered by use, then name: history, python, rust, rust-corrosion.
        // First their domains, then one request per domain with tags under it.
        let reviewer = FakeReviewer::new(vec![
            json!({"domains": [
                {"tag": 1, "domain": "history"},
                {"tag": 2, "domain": "technology"},
                {"tag": 3, "domain": "Technology"},
                {"tag": 4, "domain": "science"},
                {"tag": 9, "domain": "nowhere"}
            ]}),
            json!({"placements": [{"tag": 1, "path": ["chemistry"]}]}),
            json!({"placements": [
                {"tag": 1, "path": ["Programming Languages"]},
                {"tag": 2, "path": ["programming-languages", "systems", "too-deep"]}
            ]}),
        ]);
        let placement = place(
            &db,
            &reviewer,
            None::<&crate::reconcile::tests::NameEmbedder>,
            &ReconcileConfig::default(),
            3,
            None,
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!((placement.unplaced, placement.placed), (4, 4));
        let descriptions: Vec<&str> = placement
            .proposed
            .iter()
            .map(|p| p.description.as_str())
            .collect();
        assert_eq!(
            descriptions,
            [
                "place  (top level) ← history",
                "place  science/chemistry ← rust-corrosion (new: science, chemistry)",
                "place  technology/programming-languages ← python, rust (new: technology, programming-languages)"
            ]
        );
        assert_eq!(
            placement.proposed[2].change,
            TagChange::Place {
                parent: vec!["technology".into(), "programming-languages".into()],
                tags: vec![tag_id(&db, "python"), tag_id(&db, "rust")]
            }
        );
        // A declined placement isn't proposed again.
        db.set_tag_decision("place:history>", "declined", "user").unwrap();
        let again = FakeReviewer::new(vec![json!({"domains": [{"tag": 1, "domain": "history"}]})]);
        let placement = place(
            &db,
            &again,
            Some(&no_names()),
            &ReconcileConfig::default(),
            3,
            None,
            |_| {},
        )
        .await
        .unwrap();
        assert!(
            !placement
                .proposed
                .iter()
                .any(|p| p.description.contains("history"))
        );
    }

    #[test]
    fn similar_vectors_end_up_together() {
        let v = |x: f32, y: f32| vec![x, y];
        let order = nearest_neighbour_order(&[v(1.0, 0.0), v(0.0, 1.0), v(0.99, 0.1), v(0.1, 0.99)]);
        assert_eq!(order, [0, 2, 3, 1]);
    }
}

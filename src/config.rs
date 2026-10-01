use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

pub const EXAMPLE: &str = include_str!("../config.example.toml");

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub llm: LlmConfig,
    pub embeddings: EmbeddingsConfig,
    pub fetch: FetchConfig,
    pub tags: TagConfig,
    pub filter: FilterConfig,
    pub run: RunConfig,
    pub notes: NotesConfig,
    pub reconcile: ReconcileConfig,
}

/// The tag review that merges duplicate tags and splits ambiguous ones.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReconcileConfig {
    /// Review the tags at the end of every `import` run.
    pub at_end_of_run: bool,
    /// Model for the review; defaults to `llm.model`. A stronger model, or
    /// thinking left on, pays off here: the review is a few requests per run.
    pub model: Option<String>,
    /// Replaces `llm.extra_body` for the review, e.g. `{}` to let a thinking
    /// model think.
    pub extra_body: Option<serde_json::Map<String, serde_json::Value>>,
    /// Tags whose embeddings are at least this similar are checked for being
    /// the same tag (cosine similarity, 0 to 1).
    pub similarity: f32,
    /// Tags on at least this many pages are checked for covering different
    /// meanings.
    pub split_min_pages: usize,
    /// Pages of one tag whose embeddings fall into two groups at most this
    /// similar to each other are checked for a split.
    pub split_similarity: f32,
    /// Tag pairs per review request.
    pub batch_size: usize,
}

impl Default for ReconcileConfig {
    fn default() -> Self {
        Self {
            at_end_of_run: true,
            model: None,
            extra_body: None,
            similarity: 0.80,
            split_min_pages: 3,
            split_similarity: 0.75,
            batch_size: 20,
        }
    }
}

impl ReconcileConfig {
    /// The `[llm]` settings with the review's model and options.
    pub fn llm_config(&self, llm: &LlmConfig) -> LlmConfig {
        let mut config = llm.clone();
        if let Some(model) = &self.model {
            config.model = model.clone();
        }
        if let Some(extra_body) = &self.extra_body {
            config.extra_body = extra_body.clone();
        }
        config
    }
}

/// What a note's title shows.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TitleStyle {
    /// The page's own title, then the model's in parentheses.
    #[default]
    Both,
    /// The page's own title, as in the browser tab.
    Page,
    /// The title the model wrote.
    Summary,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NotesConfig {
    pub title: TitleStyle,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FilterMode {
    Allow,
    #[default]
    Deny,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FilterConfig {
    pub mode: FilterMode,
    /// `domain:`, `glob:`, `regex:` or `prefix:` rules; see `filter.rs`.
    pub rules: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunConfig {
    /// Pages processed at the same time.
    pub concurrency: usize,
    /// Pages fetched at the same time from one domain.
    pub per_domain: usize,
    /// The first pages of a run are processed one at a time, so the tags they
    /// create are there for the pages after them.
    pub sequential_start: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            concurrency: 4,
            per_domain: 2,
            sequential_start: 20,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddingsConfig {
    pub enabled: bool,
    pub model: String,
    /// Defaults to `llm.base_url`.
    pub base_url: Option<String>,
    /// Defaults to `llm.api_key_env`.
    pub api_key_env: Option<String>,
    /// How much of a page's text is embedded to pick the tags shown to the model.
    pub page_chars: usize,
    pub timeout_secs: u64,
    /// US dollars per million tokens.
    pub price_per_mtok: f64,
}

impl Default for EmbeddingsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: "nomic-embed-text".into(),
            base_url: None,
            api_key_env: None,
            page_chars: 2000,
            timeout_secs: 60,
            price_per_mtok: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    #[default]
    OpenaiCompatible,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StructuredOutput {
    #[default]
    JsonSchema,
    JsonObject,
    None,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmConfig {
    pub provider: Provider,
    pub base_url: String,
    pub model: String,
    pub api_key_env: Option<String>,
    pub structured_output: StructuredOutput,
    pub max_input_chars: usize,
    pub temperature: f32,
    pub timeout_secs: u64,
    /// Extra fields merged into every request body, for server-specific
    /// options such as turning off a model's thinking step.
    pub extra_body: serde_json::Map<String, serde_json::Value>,
    /// Retries after a connection error, timeout, HTTP 429 or server error.
    pub retries: u32,
    /// Limit on requests per minute, for cloud APIs; 0 means no limit.
    pub requests_per_minute: u32,
    /// US dollars per million tokens, for cost estimates and --max-cost.
    pub price_input_per_mtok: f64,
    pub price_output_per_mtok: f64,
}

impl LlmConfig {
    pub fn prices(&self) -> crate::usage::Prices {
        crate::usage::Prices {
            input_per_mtok: self.price_input_per_mtok,
            output_per_mtok: self.price_output_per_mtok,
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: Provider::default(),
            base_url: "http://localhost:11434/v1".into(),
            model: "llama3.1:8b".into(),
            api_key_env: None,
            structured_output: StructuredOutput::default(),
            max_input_chars: 12_000,
            temperature: 0.2,
            timeout_secs: 300,
            extra_body: serde_json::Map::new(),
            retries: 3,
            requests_per_minute: 0,
            price_input_per_mtok: 0.0,
            price_output_per_mtok: 0.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FetchConfig {
    pub timeout_secs: u64,
    pub user_agent: String,
    /// Retries after a timeout, connection error, HTTP 429 or server error.
    pub retries: u32,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 20,
            retries: 2,
            user_agent: "Mozilla/5.0 (compatible; tabkeeper/0.1; +https://github.com/agegechkori/tabkeeper)"
                .into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TagConfig {
    /// Levels in the tag tree built at the end of a run (phase 3).
    pub max_depth: usize,
    pub max_per_page: usize,
    pub max_new_per_page: usize,
    /// How many existing tags are shown to the model for each page.
    pub vocabulary_limit: usize,
}

impl Default for TagConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_per_page: 5,
            max_new_per_page: 3,
            vocabulary_limit: 50,
        }
    }
}

impl Config {
    /// Loads the config from `explicit` (which must exist), or else from the
    /// default location if a file is there, or else returns the defaults.
    pub fn load(explicit: Option<&Path>) -> Result<(Self, Option<PathBuf>)> {
        let path = match explicit {
            Some(p) => {
                if !p.exists() {
                    bail!("config file {} does not exist", p.display());
                }
                Some(p.to_path_buf())
            }
            None => default_path().filter(|p| p.exists()),
        };
        let config = match &path {
            Some(p) => {
                let text = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
                Self::parse(&text).with_context(|| format!("parsing {}", p.display()))?
            }
            None => Self::default(),
        };
        Ok((config, path))
    }

    pub fn parse(text: &str) -> Result<Self> {
        let config: Self = toml::from_str(text)?;
        if config.llm.timeout_secs == 0
            || config.fetch.timeout_secs == 0
            || config.embeddings.timeout_secs == 0
        {
            bail!("timeout_secs must be at least 1 in [llm], [fetch] and [embeddings]");
        }
        if config.reconcile.batch_size == 0 || config.reconcile.split_min_pages < 2 {
            bail!("reconcile.batch_size must be at least 1 and reconcile.split_min_pages at least 2");
        }
        if config.run.concurrency == 0 || config.run.per_domain == 0 {
            bail!("run.concurrency and run.per_domain must be at least 1");
        }
        if config.tags.max_depth == 0 || config.tags.max_per_page == 0 {
            bail!("tags.max_depth and tags.max_per_page must be at least 1");
        }
        Ok(config)
    }
}

pub fn default_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "tabkeeper").map(|d| d.config_dir().join("config.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_matches_defaults() {
        let parsed = Config::parse(EXAMPLE).unwrap();
        let defaults = Config::default();
        assert_eq!(parsed.llm.base_url, defaults.llm.base_url);
        assert_eq!(parsed.llm.model, defaults.llm.model);
        assert_eq!(parsed.llm.max_input_chars, defaults.llm.max_input_chars);
        assert_eq!(parsed.fetch.user_agent, defaults.fetch.user_agent);
        assert_eq!(parsed.tags.vocabulary_limit, defaults.tags.vocabulary_limit);
    }

    #[test]
    fn rejects_unknown_keys() {
        assert!(Config::parse("[llm]\nmodle = \"x\"").is_err());
    }

    #[test]
    fn partial_config_keeps_other_defaults() {
        let c = Config::parse("[llm]\nmodel = \"qwen3:8b\"").unwrap();
        assert_eq!(c.llm.model, "qwen3:8b");
        assert_eq!(c.llm.base_url, "http://localhost:11434/v1");
        assert_eq!(c.tags.max_depth, 3);
    }
}

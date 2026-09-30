use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

pub const EXAMPLE: &str = include_str!("../config.example.toml");

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub llm: LlmConfig,
    pub fetch: FetchConfig,
    pub tags: TagConfig,
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
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FetchConfig {
    pub timeout_secs: u64,
    pub user_agent: String,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 20,
            user_agent: "Mozilla/5.0 (compatible; tabkeeper/0.1; +https://github.com/agegechkori/tabkeeper)"
                .into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TagConfig {
    pub max_depth: usize,
    pub max_per_page: usize,
    pub max_new_per_page: usize,
    pub vocabulary_limit: usize,
}

impl Default for TagConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_per_page: 5,
            max_new_per_page: 2,
            vocabulary_limit: 300,
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

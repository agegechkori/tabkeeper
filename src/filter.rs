use anyhow::{Context, Result, bail};
use globset::{Glob, GlobMatcher};
use regex::Regex;
use url::Url;

use crate::config::{FilterConfig, FilterMode};

/// One URL rule: `domain:example.com` (subdomains included),
/// `glob:https://*.corp.internal/*`, `regex:^https://mail\.google\.com/` or
/// `prefix:http://localhost`.
#[derive(Debug)]
enum Rule {
    Domain(String),
    Glob(GlobMatcher),
    Regex(Regex),
    Prefix(String),
}

impl Rule {
    fn parse(text: &str) -> Result<Self> {
        let (kind, value) = text.split_once(':').with_context(|| {
            format!("filter rule {text:?} needs a kind: domain:, glob:, regex: or prefix:")
        })?;
        let value = value.trim();
        if value.is_empty() {
            bail!("filter rule {text:?} is empty");
        }
        Ok(match kind.trim() {
            "domain" => Self::Domain(
                value
                    .trim_start_matches("*.")
                    .trim_matches('.')
                    .to_ascii_lowercase(),
            ),
            "glob" => Self::Glob(
                Glob::new(value)
                    .with_context(|| format!("filter rule {text:?}"))?
                    .compile_matcher(),
            ),
            "regex" => Self::Regex(Regex::new(value).with_context(|| format!("filter rule {text:?}"))?),
            "prefix" => Self::Prefix(value.to_string()),
            other => {
                bail!("unknown filter rule kind {other:?} in {text:?}; use domain:, glob:, regex: or prefix:")
            }
        })
    }

    fn matches(&self, url: &str, host: &str) -> bool {
        match self {
            Self::Domain(domain) => host == domain || host.ends_with(&format!(".{domain}")),
            Self::Glob(glob) => glob.is_match(url),
            Self::Regex(regex) => regex.is_match(url),
            Self::Prefix(prefix) => url.starts_with(prefix.as_str()),
        }
    }
}

/// Which URLs are processed. A URL passes if it matches an allow rule (or
/// there are none) and matches no deny rule.
#[derive(Debug, Default)]
pub struct Filter {
    allow: Vec<Rule>,
    deny: Vec<Rule>,
}

impl Filter {
    /// The config's rules go on its mode's list; `--allow` and `--deny` rules
    /// from the command line are added for this run.
    pub fn new(config: &FilterConfig, extra_allow: &[String], extra_deny: &[String]) -> Result<Self> {
        let parse = |rules: &[String]| rules.iter().map(|r| Rule::parse(r)).collect::<Result<Vec<_>>>();
        let mut filter = Self {
            allow: parse(extra_allow)?,
            deny: parse(extra_deny)?,
        };
        let configured = parse(&config.rules)?;
        match config.mode {
            FilterMode::Allow => {
                if configured.is_empty() {
                    bail!("filter.mode = \"allow\" needs at least one rule in filter.rules");
                }
                filter.allow.extend(configured);
            }
            FilterMode::Deny => filter.deny.extend(configured),
        }
        Ok(filter)
    }

    pub fn allows(&self, url: &str) -> bool {
        let host = Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
            .unwrap_or_default();
        (self.allow.is_empty() || self.allow.iter().any(|r| r.matches(url, &host)))
            && !self.deny.iter().any(|r| r.matches(url, &host))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter(mode: FilterMode, rules: &[&str], allow: &[&str], deny: &[&str]) -> Result<Filter> {
        let strings = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        Filter::new(
            &FilterConfig {
                mode,
                rules: strings(rules),
            },
            &strings(allow),
            &strings(deny),
        )
    }

    #[test]
    fn deny_rules_of_each_kind() {
        let f = filter(
            FilterMode::Deny,
            &[
                "domain:bank.example.com",
                "glob:https://*.corp.internal/*",
                "regex:^https://mail\\.google\\.com/",
                "prefix:http://localhost",
            ],
            &[],
            &[],
        )
        .unwrap();
        assert!(!f.allows("https://bank.example.com/login"));
        assert!(
            !f.allows("https://www.bank.example.com/"),
            "subdomains are included"
        );
        assert!(
            f.allows("https://notbank.example.com/"),
            "only whole labels match"
        );
        assert!(!f.allows("https://wiki.corp.internal/page"));
        assert!(!f.allows("https://mail.google.com/mail/u/0/"));
        assert!(!f.allows("http://localhost:3000/"));
        assert!(f.allows("https://en.wikipedia.org/wiki/Rust"));
    }

    #[test]
    fn allow_mode_and_command_line_rules() {
        let f = filter(
            FilterMode::Allow,
            &["domain:wikipedia.org"],
            &[],
            &["prefix:https://en.wikipedia.org/wiki/Special:"],
        )
        .unwrap();
        assert!(f.allows("https://en.wikipedia.org/wiki/Rust"));
        assert!(
            !f.allows("https://en.wikipedia.org/wiki/Special:Random"),
            "deny wins over allow"
        );
        assert!(!f.allows("https://github.com/"));

        let f = filter(FilterMode::Deny, &[], &["domain:github.com"], &[]).unwrap();
        assert!(f.allows("https://github.com/rust-lang/rust"));
        assert!(
            !f.allows("https://example.com/"),
            "an --allow rule limits the run to matching URLs"
        );

        assert!(
            filter(FilterMode::Deny, &[], &[], &[])
                .unwrap()
                .allows("https://anything.example/")
        );
    }

    #[test]
    fn rejects_bad_rules() {
        assert!(
            filter(FilterMode::Allow, &[], &[], &[]).is_err(),
            "allow mode without rules would skip everything"
        );
        assert!(filter(FilterMode::Deny, &["example.com"], &[], &[]).is_err());
        assert!(filter(FilterMode::Deny, &["host:example.com"], &[], &[]).is_err());
        assert!(filter(FilterMode::Deny, &["regex:("], &[], &[]).is_err());
        assert!(filter(FilterMode::Deny, &["domain:"], &[], &[]).is_err());
    }
}

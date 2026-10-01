use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// What one stage of a run (summaries or embeddings) asked of its server.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StageUsage {
    /// HTTP requests sent, including retries.
    pub requests: u64,
    /// Requests sent again after a connection error, timeout, 429 or 5xx.
    pub retries: u64,
    /// Summaries asked for again because the reply wasn't valid.
    pub invalid_replies: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Successful responses that didn't say how many tokens they used.
    pub responses_without_usage: u64,
    /// Texts turned into vectors (embeddings only).
    pub texts_embedded: u64,
    /// Time spent waiting for responses, added up over all requests; with
    /// several pages at once this exceeds the run's wall time.
    pub request_secs: f64,
}

impl StageUsage {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// Records a successful response's `usage` field, if it has one.
    pub fn record_response(&mut self, body: &Value, elapsed: Duration) {
        self.request_secs += elapsed.as_secs_f64();
        match parse_usage(body) {
            Some((input, output)) => {
                self.input_tokens += input;
                self.output_tokens += output;
            }
            None => self.responses_without_usage += 1,
        }
    }
}

/// (input, output) tokens from an OpenAI-style `usage` object. Embeddings
/// responses only report input (`prompt_tokens`).
pub fn parse_usage(body: &Value) -> Option<(u64, u64)> {
    let usage = body.get("usage")?;
    let input = usage.get("prompt_tokens")?.as_u64()?;
    let output = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Some((input, output))
}

/// A usage counter a client updates as it goes, shared with whoever reports it.
#[derive(Debug, Default)]
pub struct UsageCounter(Mutex<StageUsage>);

impl UsageCounter {
    pub fn update(&self, f: impl FnOnce(&mut StageUsage)) {
        f(&mut self.0.lock().expect("usage lock is never poisoned"));
    }

    pub fn snapshot(&self) -> StageUsage {
        self.0.lock().expect("usage lock is never poisoned").clone()
    }
}

/// Prices in US dollars per million tokens; zero means not set.
#[derive(Debug, Clone, Copy, Default)]
pub struct Prices {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
}

impl Prices {
    pub fn is_set(&self) -> bool {
        self.input_per_mtok > 0.0 || self.output_per_mtok > 0.0
    }

    pub fn cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        (input_tokens as f64 * self.input_per_mtok + output_tokens as f64 * self.output_per_mtok) / 1e6
    }
}

/// Whether a server URL points at this machine, where tokens cost nothing.
pub fn is_local(base_url: &str) -> bool {
    url::Url::parse(base_url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| matches!(host.as_str(), "localhost" | "127.0.0.1" | "[::1]" | "::1"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn records_reported_and_missing_usage() {
        let mut usage = StageUsage::default();
        usage.record_response(
            &json!({"usage": {"prompt_tokens": 100, "completion_tokens": 20}}),
            Duration::from_secs(2),
        );
        usage.record_response(
            &json!({"usage": {"prompt_tokens": 7, "total_tokens": 7}}),
            Duration::from_secs(1),
        );
        usage.record_response(&json!({"choices": []}), Duration::from_secs(1));
        assert_eq!(
            (usage.input_tokens, usage.output_tokens, usage.total_tokens()),
            (107, 20, 127)
        );
        assert_eq!(usage.responses_without_usage, 1);
        assert_eq!(usage.request_secs, 4.0);
    }

    #[test]
    fn costs_and_local_servers() {
        let prices = Prices {
            input_per_mtok: 3.0,
            output_per_mtok: 15.0,
        };
        assert!((prices.cost(1_000_000, 100_000) - 4.5).abs() < 1e-9);
        assert!(!Prices::default().is_set());
        assert!(is_local("http://localhost:11434/v1"));
        assert!(is_local("http://127.0.0.1:1234/v1"));
        assert!(!is_local("https://api.openai.com/v1"));
    }
}

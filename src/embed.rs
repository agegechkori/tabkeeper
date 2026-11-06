use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::config::{EmbeddingsConfig, LlmConfig};
use crate::usage::{StageUsage, UsageCounter};

/// Texts sent to the server per embeddings request.
pub const BATCH_SIZE: usize = 64;

pub trait Embedder {
    /// Model name, stored with each vector so vectors from different models
    /// are never compared.
    fn model(&self) -> &str;
    /// One unit-length vector per input text, in order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Requests and tokens so far.
    fn usage(&self) -> StageUsage {
        StageUsage::default()
    }
}

/// Client for the `/embeddings` endpoint of OpenAI-compatible servers
/// (Ollama, LM Studio, llama.cpp, vLLM, OpenAI).
pub struct OpenAiEmbedder {
    client: reqwest::Client,
    url: String,
    model: String,
    api_key: Option<String>,
    usage: UsageCounter,
}

impl OpenAiEmbedder {
    pub fn new(config: &EmbeddingsConfig, llm: &LlmConfig) -> Result<Self> {
        let base_url = config.base_url.as_deref().unwrap_or(&llm.base_url);
        let api_key = match config.api_key_env.as_ref().or(llm.api_key_env.as_ref()) {
            Some(var) => {
                Some(std::env::var(var).with_context(|| format!("environment variable {var} is not set"))?)
            }
            None => None,
        };
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()?;
        Ok(Self {
            client,
            url: format!("{}/embeddings", base_url.trim_end_matches('/')),
            model: config.model.clone(),
            api_key,
            usage: UsageCounter::default(),
        })
    }
}

impl Embedder for OpenAiEmbedder {
    fn model(&self) -> &str {
        &self.model
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.usage.update(|u| u.requests += 1);
        let started = std::time::Instant::now();
        let mut request = self
            .client
            .post(&self.url)
            .json(&json!({"model": self.model, "input": texts}));
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request
            .send()
            .await
            .with_context(|| format!("calling {}", self.url))?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            bail!(
                "embeddings server returned HTTP {status}: {}",
                crate::extract::truncate_chars(text.trim(), 300)
            );
        }
        let vectors = parse_embeddings(&text, texts.len())?;
        if let Ok(value) = serde_json::from_str::<Value>(&text) {
            self.usage.update(|u| {
                u.record_response(&value, started.elapsed());
                u.texts_embedded += texts.len() as u64;
            });
        }
        Ok(vectors)
    }

    fn usage(&self) -> StageUsage {
        self.usage.snapshot()
    }
}

fn parse_embeddings(body: &str, expected: usize) -> Result<Vec<Vec<f32>>> {
    let value: Value = serde_json::from_str(body).context("invalid embeddings response")?;
    let data = value["data"]
        .as_array()
        .ok_or_else(|| anyhow!("embeddings response has no data"))?;
    if data.len() != expected {
        bail!("asked for {expected} embeddings, got {}", data.len());
    }
    let mut out = vec![Vec::new(); expected];
    for (position, item) in data.iter().enumerate() {
        let index = item["index"].as_u64().map_or(position, |i| i as usize);
        let vector: Vec<f32> = item["embedding"]
            .as_array()
            .ok_or_else(|| anyhow!("embedding {index} is not a list of numbers"))?
            .iter()
            .map(|x| {
                x.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| anyhow!("embedding {index} has a non-number"))
            })
            .collect::<Result<_>>()?;
        *out.get_mut(index)
            .ok_or_else(|| anyhow!("embedding index {index} out of range"))? = normalized(vector);
    }
    Ok(out)
}

fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

/// Cosine similarity of two unit-length vectors.
pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_normalizes_out_of_order_response() {
        let body = r#"{"data":[{"index":1,"embedding":[0,2]},{"index":0,"embedding":[3,4]}]}"#;
        let vectors = parse_embeddings(body, 2).unwrap();
        assert_eq!(vectors, [vec![0.6, 0.8], vec![0.0, 1.0]]);
    }

    #[test]
    fn rejects_wrong_count() {
        assert!(parse_embeddings(r#"{"data":[]}"#, 1).is_err());
        assert!(parse_embeddings(r#"{"error":"model not found"}"#, 1).is_err());
    }
}

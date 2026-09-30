use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::config::{EmbeddingsConfig, LlmConfig};

/// Texts sent to the server per embeddings request.
pub const BATCH_SIZE: usize = 64;

pub trait Embedder {
    /// Model name, stored with each vector so vectors from different models
    /// are never compared.
    fn model(&self) -> &str;
    /// One unit-length vector per input text, in order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// Client for the `/embeddings` endpoint of OpenAI-compatible servers
/// (Ollama, LM Studio, llama.cpp, vLLM, OpenAI).
pub struct OpenAiEmbedder {
    client: reqwest::Client,
    url: String,
    model: String,
    api_key: Option<String>,
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
        })
    }
}

impl Embedder for OpenAiEmbedder {
    fn model(&self) -> &str {
        &self.model
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
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
        parse_embeddings(&text, texts.len())
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

/// Tag vectors kept in memory for similarity search. A brute-force scan is
/// fast enough for tens of thousands of tags.
#[derive(Default)]
pub struct TagIndex {
    vectors: Vec<(i64, Vec<f32>)>,
}

impl TagIndex {
    pub fn new(vectors: Vec<(i64, Vec<f32>)>) -> Self {
        Self { vectors }
    }

    pub fn insert(&mut self, tag_id: i64, vector: Vec<f32>) {
        self.vectors.retain(|(id, _)| *id != tag_id);
        self.vectors.push((tag_id, vector));
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Tag ids ordered by similarity to `query`, most similar first, among
    /// the tags `keep` accepts.
    pub fn nearest(&self, query: &[f32], limit: usize, keep: impl Fn(i64) -> bool) -> Vec<i64> {
        let mut scored: Vec<(f32, i64)> = self
            .vectors
            .iter()
            .filter(|(id, v)| v.len() == query.len() && keep(*id))
            .map(|(id, v)| (similarity(query, v), *id))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.into_iter().take(limit).map(|(_, id)| id).collect()
    }
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

    #[test]
    fn nearest_orders_by_similarity() {
        let mut index = TagIndex::new(vec![(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0])]);
        index.insert(3, normalized(vec![1.0, 1.0]));
        index.insert(1, vec![1.0, 0.0]); // replacing keeps one entry per tag
        assert_eq!(index.len(), 3);
        assert_eq!(index.nearest(&[1.0, 0.0], 2, |_| true), [1, 3]);
        assert_eq!(index.nearest(&[1.0, 0.0], 5, |id| id != 1), [3, 2]);
        // Vectors from a model with other dimensions are never compared.
        assert!(index.nearest(&[1.0, 0.0, 0.0], 5, |_| true).is_empty());
    }
}

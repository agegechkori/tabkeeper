use std::time::Duration;

use anyhow::{Result, bail};

use crate::config::FetchConfig;

/// Pages larger than this are cut before extraction.
const MAX_HTML_BYTES: usize = 5_000_000;

pub struct FetchedPage {
    pub final_url: String,
    pub html: String,
}

pub trait Fetcher {
    async fn fetch(&self, url: &str) -> Result<FetchedPage>;
}

pub struct HttpFetcher {
    client: reqwest::Client,
}

impl HttpFetcher {
    pub fn new(config: &FetchConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(&config.user_agent)
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()?;
        Ok(Self { client })
    }
}

impl Fetcher for HttpFetcher {
    async fn fetch(&self, url: &str) -> Result<FetchedPage> {
        let response = self.client.get(url).send().await?;
        let status = response.status();
        if !status.is_success() {
            bail!("HTTP {status}");
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !content_type.is_empty() && !content_type.contains("html") && !content_type.contains("xml") {
            bail!("unsupported content type: {content_type}");
        }
        let final_url = response.url().to_string();
        let mut html = response.text().await?;
        if html.len() > MAX_HTML_BYTES {
            let mut cut = MAX_HTML_BYTES;
            while !html.is_char_boundary(cut) {
                cut -= 1;
            }
            html.truncate(cut);
        }
        Ok(FetchedPage { final_url, html })
    }
}

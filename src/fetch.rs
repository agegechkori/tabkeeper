use std::fmt;
use std::time::Duration;

use anyhow::Result;
use reqwest::StatusCode;

use crate::config::FetchConfig;

/// Pages larger than this are cut before extraction.
const MAX_HTML_BYTES: usize = 5_000_000;
/// Longest wait between retries, whatever a server's Retry-After asks for.
const MAX_RETRY_WAIT: Duration = Duration::from_secs(30);

pub struct FetchedPage {
    pub final_url: String,
    pub html: String,
}

/// Why a page couldn't be fetched. Stored with the page for the final report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchErrorKind {
    /// HTTP 404 or 410.
    NotFound,
    /// HTTP 401, 403 or 429: a login wall or bot protection.
    Blocked,
    /// Any other HTTP error status.
    HttpError,
    /// DNS failure, refused or reset connection, TLS error.
    Connection,
    Timeout,
    /// The page loaded but isn't HTML, e.g. a PDF.
    UnsupportedType,
}

impl FetchErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotFound => "not_found",
            Self::Blocked => "blocked",
            Self::HttpError => "http_error",
            Self::Connection => "connection",
            Self::Timeout => "timeout",
            Self::UnsupportedType => "unsupported_type",
        }
    }

    /// Whether the page itself couldn't be reached, as opposed to reached
    /// but not usable. Unreachable pages get a stub note.
    pub fn is_unreachable(self) -> bool {
        self != Self::UnsupportedType
    }
}

#[derive(Debug)]
pub struct FetchError {
    pub kind: FetchErrorKind,
    pub message: String,
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FetchError {}

pub trait Fetcher {
    async fn fetch(&self, url: &str) -> Result<FetchedPage, FetchError>;
}

pub struct HttpFetcher {
    client: reqwest::Client,
    retries: u32,
}

impl HttpFetcher {
    pub fn new(config: &FetchConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(&config.user_agent)
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()?;
        Ok(Self {
            client,
            retries: config.retries,
        })
    }

    async fn fetch_once(&self, url: &str) -> Result<FetchedPage, Attempt> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| Attempt::from_reqwest(&e))?;
        let status = response.status();
        if !status.is_success() {
            return Err(Attempt::from_status(status, retry_after(&response)));
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !content_type.is_empty() && !content_type.contains("html") && !content_type.contains("xml") {
            return Err(Attempt::fatal(
                FetchErrorKind::UnsupportedType,
                format!("unsupported content type: {content_type}"),
            ));
        }
        let final_url = response.url().to_string();
        let mut bytes = response
            .bytes()
            .await
            .map_err(|e| Attempt::from_reqwest(&e))?
            .to_vec();
        bytes.truncate(MAX_HTML_BYTES);
        let html = decode_html(&bytes, header_charset(&content_type));
        Ok(FetchedPage { final_url, html })
    }
}

impl Fetcher for HttpFetcher {
    async fn fetch(&self, url: &str) -> Result<FetchedPage, FetchError> {
        let mut attempt = 0;
        loop {
            match self.fetch_once(url).await {
                Ok(page) => return Ok(page),
                Err(failed) if failed.retry && attempt < self.retries => {
                    tokio::time::sleep(
                        failed
                            .wait
                            .unwrap_or_else(|| backoff(attempt))
                            .min(MAX_RETRY_WAIT),
                    )
                    .await;
                    attempt += 1;
                }
                Err(failed) => return Err(failed.error),
            }
        }
    }
}

/// The charset named in a Content-Type value, e.g. `text/html; charset=utf-8`.
fn header_charset(content_type: &str) -> Option<&str> {
    let (_, rest) = content_type.split_once("charset=")?;
    Some(
        rest.split(';')
            .next()
            .unwrap_or(rest)
            .trim()
            .trim_matches(['"', '\'']),
    )
}

/// Decodes a page: a byte order mark wins, then the charset from the HTTP
/// header, then one declared in the page's `<meta>` tags, then UTF-8.
/// Invalid bytes become replacement characters instead of failing the page.
pub fn decode_html(bytes: &[u8], header_charset: Option<&str>) -> String {
    let declared = header_charset
        .and_then(|label| encoding_rs::Encoding::for_label(label.as_bytes()))
        .or_else(|| meta_charset(&bytes[..bytes.len().min(4096)]));
    let (text, _, _) = declared.unwrap_or(encoding_rs::UTF_8).decode(bytes);
    text.into_owned()
}

/// `<meta charset="…">` or `<meta http-equiv="Content-Type" content="…; charset=…">`
/// near the start of the page.
fn meta_charset(head: &[u8]) -> Option<&'static encoding_rs::Encoding> {
    let head = String::from_utf8_lossy(head).to_ascii_lowercase();
    let start = head.find("<meta")?;
    let at = start + head[start..].find("charset=")? + "charset=".len();
    let label: String = head[at..]
        .trim_start_matches(['"', '\'', ' '])
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ':' | '.'))
        .collect();
    // A page whose bytes were read as ASCII to find this tag can't really be
    // UTF-16; the HTML standard treats such a declaration as UTF-8.
    encoding_rs::Encoding::for_label(label.as_bytes()).map(|e| e.output_encoding())
}

/// 1 s, 2 s, 4 s, …
pub fn backoff(attempt: u32) -> Duration {
    Duration::from_secs(1 << attempt.min(5))
}

/// The server's Retry-After header, when given in seconds.
pub fn retry_after(response: &reqwest::Response) -> Option<Duration> {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

/// One failed attempt, and whether trying again might help.
struct Attempt {
    error: FetchError,
    retry: bool,
    wait: Option<Duration>,
}

impl Attempt {
    fn fatal(kind: FetchErrorKind, message: String) -> Self {
        Self {
            error: FetchError { kind, message },
            retry: false,
            wait: None,
        }
    }

    fn from_status(status: StatusCode, wait: Option<Duration>) -> Self {
        let kind = classify_status(status);
        let retry = status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
        Self {
            error: FetchError {
                kind,
                message: format!("HTTP {status}"),
            },
            retry,
            wait,
        }
    }

    fn from_reqwest(err: &reqwest::Error) -> Self {
        // reqwest's own message repeats the URL; the innermost cause is the
        // useful part, e.g. "failed to lookup address information: …".
        let mut chain = vec![err.to_string()];
        let mut source = std::error::Error::source(err);
        while let Some(s) = source {
            chain.push(s.to_string());
            source = s.source();
        }
        let detail = chain.last().cloned().unwrap_or_default();
        let (kind, message, retry) = if err.is_timeout() {
            (FetchErrorKind::Timeout, "timed out".to_string(), true)
        } else if chain.iter().any(|m| m.contains("dns error")) {
            // A name that doesn't resolve won't resolve a second later.
            (
                FetchErrorKind::Connection,
                format!("DNS lookup failed: {detail}"),
                false,
            )
        } else if err.is_connect() {
            (
                FetchErrorKind::Connection,
                format!("could not connect: {detail}"),
                true,
            )
        } else {
            (
                FetchErrorKind::Connection,
                format!("connection failed: {detail}"),
                true,
            )
        };
        Self {
            error: FetchError { kind, message },
            retry,
            wait: None,
        }
    }
}

fn classify_status(status: StatusCode) -> FetchErrorKind {
    match status.as_u16() {
        404 | 410 => FetchErrorKind::NotFound,
        401 | 403 | 429 => FetchErrorKind::Blocked,
        _ => FetchErrorKind::HttpError,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_statuses() {
        assert_eq!(classify_status(StatusCode::NOT_FOUND), FetchErrorKind::NotFound);
        assert_eq!(classify_status(StatusCode::GONE), FetchErrorKind::NotFound);
        assert_eq!(classify_status(StatusCode::FORBIDDEN), FetchErrorKind::Blocked);
        assert_eq!(
            classify_status(StatusCode::TOO_MANY_REQUESTS),
            FetchErrorKind::Blocked
        );
        assert_eq!(
            classify_status(StatusCode::BAD_GATEWAY),
            FetchErrorKind::HttpError
        );
        assert!(Attempt::from_status(StatusCode::TOO_MANY_REQUESTS, None).retry);
        assert!(Attempt::from_status(StatusCode::SERVICE_UNAVAILABLE, None).retry);
        assert!(!Attempt::from_status(StatusCode::NOT_FOUND, None).retry);
        assert!(FetchErrorKind::Timeout.is_unreachable());
        assert!(!FetchErrorKind::UnsupportedType.is_unreachable());
    }

    #[test]
    fn decodes_using_header_then_meta_then_utf8() {
        let (cyrillic, _, _) = encoding_rs::WINDOWS_1251.encode("Привет, мир");
        let page = [
            b"<html><head><meta charset=\"windows-1251\"></head><body>".as_slice(),
            &cyrillic,
            b"</body>",
        ]
        .concat();
        assert!(
            decode_html(&page, None).contains("Привет, мир"),
            "meta charset is used"
        );

        let page = [
            b"<meta http-equiv=\"Content-Type\" content=\"text/html; charset=windows-1251\">".as_slice(),
            &cyrillic,
        ]
        .concat();
        assert!(
            decode_html(&page, None).contains("Привет, мир"),
            "http-equiv form"
        );

        let (shift_jis, _, _) = encoding_rs::SHIFT_JIS.encode("こんにちは");
        let page = [b"<meta charset=\"windows-1251\">".as_slice(), &shift_jis].concat();
        assert!(
            decode_html(&page, Some("shift_jis")).contains("こんにちは"),
            "the header wins"
        );

        assert_eq!(decode_html("héllo".as_bytes(), None), "héllo", "UTF-8 by default");
        assert_eq!(
            decode_html("<meta charset=\"utf-16\">héllo".as_bytes(), None),
            "<meta charset=\"utf-16\">héllo",
            "a UTF-16 declaration in <meta> means UTF-8"
        );
        assert_eq!(
            decode_html(b"\xffbad", Some("utf-8")),
            "\u{fffd}bad",
            "invalid bytes don't fail the page"
        );
    }

    #[test]
    fn reads_charset_from_content_type() {
        assert_eq!(header_charset("text/html; charset=utf-8"), Some("utf-8"));
        assert_eq!(
            header_charset("text/html; charset=\"iso-8859-1\"; foo=bar"),
            Some("iso-8859-1")
        );
        assert_eq!(header_charset("text/html"), None);
    }

    #[test]
    fn backoff_grows_and_is_capped() {
        assert_eq!(backoff(0), Duration::from_secs(1));
        assert_eq!(backoff(2), Duration::from_secs(4));
        assert_eq!(backoff(40), Duration::from_secs(32));
    }
}

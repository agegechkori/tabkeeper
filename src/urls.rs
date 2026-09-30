use url::Url;

/// Query parameters that only track where a click came from. Removing them
/// lets the same page opened from different links deduplicate to one note.
const TRACKING_PARAMS: &[&str] = &[
    "fbclid", "gclid", "dclid", "gbraid", "wbraid", "msclkid", "yclid", "twclid", "igshid", "mc_cid",
    "mc_eid", "_hsenc", "_hsmi",
];

#[derive(Debug, PartialEq, Eq)]
pub enum Rejected {
    /// Browser-internal, local or otherwise non-web URL (chrome://, file://, …).
    NotWeb(String),
    Invalid,
}

/// Normalizes a URL for deduplication: web schemes only, no fragment, no
/// tracking parameters.
pub fn normalize(raw: &str) -> Result<String, Rejected> {
    let mut url = Url::parse(raw.trim()).map_err(|_| Rejected::Invalid)?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(Rejected::NotWeb(url.scheme().to_string()));
    }
    url.set_fragment(None);

    let kept: Vec<(String, String)> = url
        .query_pairs()
        .filter(|(k, _)| !is_tracking_param(k))
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();
    if url.query().is_some() {
        if kept.is_empty() {
            url.set_query(None);
        } else if kept.len() != url.query_pairs().count() {
            url.query_pairs_mut().clear().extend_pairs(kept);
        }
    }
    Ok(url.into())
}

fn is_tracking_param(key: &str) -> bool {
    key.starts_with("utm_") || TRACKING_PARAMS.contains(&key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_fragment_and_tracking() {
        assert_eq!(
            normalize("https://Example.com/a?utm_source=x&id=5&fbclid=abc#top").unwrap(),
            "https://example.com/a?id=5"
        );
        assert_eq!(
            normalize("https://example.com/a?utm_medium=y").unwrap(),
            "https://example.com/a"
        );
    }

    #[test]
    fn leaves_untracked_query_untouched() {
        assert_eq!(
            normalize("https://example.com/search?q=a+b&page=2").unwrap(),
            "https://example.com/search?q=a+b&page=2"
        );
    }

    #[test]
    fn rejects_non_web() {
        assert_eq!(
            normalize("chrome://settings"),
            Err(Rejected::NotWeb("chrome".into()))
        );
        assert_eq!(
            normalize("file:///etc/hosts"),
            Err(Rejected::NotWeb("file".into()))
        );
        assert_eq!(normalize("not a url"), Err(Rejected::Invalid));
    }
}

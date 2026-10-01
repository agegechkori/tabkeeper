use dom_smoothie::{Config, Readability, TextMode};

pub struct Extracted {
    pub title: String,
    /// Main text of the page, whitespace-cleaned and cut to the requested
    /// length. Empty if no readable content was found.
    pub text: String,
    /// Whether the text was longer than the requested length and was cut.
    pub truncated: bool,
}

/// Extracts the readable title and text of an HTML page.
pub fn extract(html: &str, url: &str, max_chars: usize) -> Extracted {
    let config = Config {
        text_mode: TextMode::Formatted,
        ..Default::default()
    };
    let Ok(mut readability) = Readability::new(html, Some(url), Some(config)) else {
        return Extracted {
            title: String::new(),
            text: String::new(),
            truncated: false,
        };
    };
    match readability.parse() {
        Ok(article) => {
            let full = clean_text(&article.text_content);
            let text = truncate_chars(&full, max_chars);
            Extracted {
                title: clean_line(&article.title),
                truncated: text.len() < full.len(),
                text,
            }
        }
        Err(_) => Extracted {
            title: clean_line(&readability.get_article_title()),
            text: String::new(),
            truncated: false,
        },
    }
}

fn clean_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Collapses runs of spaces within lines and runs of blank lines.
fn clean_text(s: &str) -> String {
    let mut out = String::new();
    let mut blank = false;
    for line in s.lines() {
        let line = clean_line(line);
        if line.is_empty() {
            blank = !out.is_empty();
            continue;
        }
        if blank {
            out.push('\n');
            blank = false;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&line);
    }
    out
}

pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        Some((i, _)) => s[..i].to_string(),
        None => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARTICLE: &str = r#"<!doctype html><html lang="en"><head><title>Rust ownership explained | Blog</title></head>
<body><nav><a href="/">Home</a> <a href="/about">About</a></nav>
<article><h1>Rust ownership explained</h1>
<p>Ownership is Rust's most unique feature, and it enables Rust to make memory safety guarantees without needing a garbage collector.</p>
<p>In this post we look at what ownership is, how borrowing works, and why the borrow checker rejects some programs that look correct at first glance.</p>
<p>Each value in Rust has a variable that is called its owner. There can only be one owner at a time. When the owner goes out of scope, the value will be dropped.</p>
</article><footer>Copyright 2026</footer></body></html>"#;

    #[test]
    fn extracts_article_text() {
        let e = extract(ARTICLE, "https://blog.example.com/ownership", 10_000);
        assert!(e.title.contains("Rust ownership"), "title: {}", e.title);
        assert!(e.text.contains("borrow checker"), "text: {}", e.text);
        assert!(!e.text.contains("Copyright"), "text: {}", e.text);
        assert!(!e.truncated);
        assert!(extract(ARTICLE, "https://blog.example.com/ownership", 50).truncated);
    }

    #[test]
    fn truncates_on_char_boundary() {
        assert_eq!(truncate_chars("héllo", 2), "hé");
        assert_eq!(truncate_chars("hi", 10), "hi");
    }

    #[test]
    fn cleans_whitespace() {
        assert_eq!(clean_text("  a   b \n\n\n c\n"), "a b\n\nc");
    }
}

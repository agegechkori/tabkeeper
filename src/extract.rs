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

/// The page's `<title>`, as a browser tab shows it.
pub fn html_title(html: &str) -> Option<String> {
    // ASCII lowercasing keeps byte positions, so they index `html` too.
    let lower = html.to_ascii_lowercase();
    // Only the document's own title, in <head>: inline SVG icons in the body
    // have <title>s too ("Menu", "Close").
    let head_end = [find_tag(&lower, "</head"), find_tag(&lower, "<body")]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(lower.len());
    let start = lower[..head_end].find("<title")?;
    let content = start + lower[start..].find('>')? + 1;
    let end = content + lower[content..].find("</title")?;
    let title = clean_line(&decode_entities(&html[content..end]));
    (!title.is_empty()).then_some(title)
}

/// Where a tag such as `</head` starts, as a whole tag name: not `</header`.
fn find_tag(lower: &str, tag: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(at) = lower[from..].find(tag).map(|i| from + i) {
        let next = lower[at + tag.len()..].chars().next();
        if next.is_none_or(|c| c == '>' || c == '/' || c.is_whitespace()) {
            return Some(at);
        }
        from = at + tag.len();
    }
    None
}

/// Decodes the HTML character references common in titles.
fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let decoded = rest.find(';').filter(|&semi| semi <= 10).and_then(|semi| {
            let name = &rest[1..semi];
            let c = match name {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some(' '),
                _ => name
                    .strip_prefix("#x")
                    .or_else(|| name.strip_prefix("#X"))
                    .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                    .or_else(|| name.strip_prefix('#').and_then(|dec| dec.parse().ok()))
                    .and_then(char::from_u32),
            }?;
            Some((c, semi + 1))
        });
        match decoded {
            Some((c, len)) => {
                out.push(c);
                rest = &rest[len..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
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
    fn reads_the_html_title() {
        assert_eq!(
            html_title(ARTICLE).as_deref(),
            Some("Rust ownership explained | Blog")
        );
        assert_eq!(
            html_title("<HTML><Title lang=en>\n  Tom &amp; Jerry &#8211; &quot;Cartoons&quot; &#x2014; ok &bogus;\n</TITLE>").as_deref(),
            Some("Tom & Jerry – \"Cartoons\" — ok &bogus;")
        );
        assert_eq!(html_title("<title>  </title>"), None);
        let icon_first =
            "<html><head><meta charset=utf-8></head><body><svg><title>Menu</title></svg></body></html>";
        assert_eq!(html_title(icon_first), None, "an icon's title is not the page's");
        let both = "<head><title>Real Title</title></head><body><svg><title>Menu</title></svg></body>";
        assert_eq!(html_title(both).as_deref(), Some("Real Title"));
        // No </head> (HTML5 allows that): </header> must not be mistaken for it.
        let no_head_end =
            "<head><meta charset=utf-8><body><header><svg><title>Menu</title></svg></header></body>";
        assert_eq!(html_title(no_head_end), None);
        let header_first = "<head><title>Real</title><header>x</header></head>";
        assert_eq!(html_title(header_first).as_deref(), Some("Real"));
        assert_eq!(html_title("<p>no title</p>"), None);
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

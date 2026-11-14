//! Reads Markdown on stdin and writes the HTML body on stdout.

use std::io::Read;

use pulldown_cmark::{Options, Parser, html};

fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).expect("reading stdin");
    let options = Options::ENABLE_TABLES
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_HEADING_ATTRIBUTES
        | Options::ENABLE_GFM;
    let mut out = String::new();
    html::push_html(&mut out, Parser::new_ext(&text, options));
    print!("{out}");
}

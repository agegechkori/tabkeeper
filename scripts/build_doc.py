"""Turn a Markdown document from docs/ into a page to read in a browser.

    python3 scripts/build_doc.py [docs/tagging-algorithm.md] [--out DIR] [--serve [PORT]]

The page has a contents sidebar, light and dark themes, and renders the
document's ```mermaid blocks as diagrams. It is written to DIR/index.html
(default: target/tagging-doc for the tagging document, target/<name>-doc
otherwise). With --serve, it is then served on http://127.0.0.1:PORT/
(default 8787) until Ctrl-C; only this machine can reach it.

Markdown is converted by scripts/md2html, a small Rust tool built on first
use (cargo, into target/md2html). Mermaid is downloaded once, pinned by
version and hash, next to the page."""
import argparse
import hashlib
import html
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"
MERMAID_SHA256 = "a43bc1afd446f9c4cc66ac5dd45d02e8d65e26fc5344ec0ef787f88d6ddb6f9e"
ALERTS = {"note": "Note", "tip": "Tip", "important": "Important", "warning": "Warning", "caution": "Caution"}


def markdown_to_html(text):
    env = dict(os.environ, CARGO_TARGET_DIR=os.path.join(ROOT, "target", "md2html"))
    manifest = os.path.join(ROOT, "scripts", "md2html", "Cargo.toml")
    subprocess.run(["cargo", "build", "--release", "--quiet", "--manifest-path", manifest], env=env, check=True)
    tool = os.path.join(ROOT, "target", "md2html", "release", "md2html")
    return subprocess.run([tool], input=text, capture_output=True, text=True, check=True).stdout


def ensure_mermaid(out_dir):
    path = os.path.join(out_dir, "mermaid.min.js")
    if os.path.exists(path) and hashlib.sha256(open(path, "rb").read()).hexdigest() == MERMAID_SHA256:
        return
    print(f"Downloading {MERMAID_URL}")
    data = urllib.request.urlopen(MERMAID_URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != MERMAID_SHA256:
        sys.exit(f"{MERMAID_URL} doesn't match the pinned hash; not using it")
    with open(path, "wb") as f:
        f.write(data)


def build_body(body):
    """Diagrams, alert labels, scrolling tables, heading anchors; returns the
    body, its title (the first heading) and the contents entries."""
    # Mermaid reads the element's text, so the escaped source works as is.
    body = re.sub(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        lambda m: f'<figure class="diagram"><pre class="mermaid">{m.group(1)}</pre></figure>',
        body,
        flags=re.S,
    )
    body = re.sub(
        r'<blockquote class="markdown-alert-(\w+)">',
        lambda m: f'<blockquote class="alert alert-{m.group(1)}"><p class="alert-title">'
        f"{ALERTS.get(m.group(1), m.group(1))}</p>",
        body,
    )
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")

    toc, used = [], set()

    def slug(text):
        s = html.unescape(re.sub(r"<[^>]+>", "", text)).lower()
        s = re.sub(r"[\s_]+", "-", re.sub(r"[^\w\s-]", "", s).strip()) or "section"
        base, n = s, 2
        while s in used:
            s, n = f"{base}-{n}", n + 1
        used.add(s)
        return s

    def heading(m):
        level, inner = int(m.group(1)), m.group(2)
        ident = slug(inner)
        if level in (2, 3):
            toc.append((level, ident, re.sub(r"<[^>]+>", "", inner)))
        return f'<h{level} id="{ident}"><a class="anchor" href="#{ident}" aria-hidden="true">#</a>{inner}</h{level}>'

    body = re.sub(r"<h([1-4])>(.*?)</h\1>", heading, body, flags=re.S)
    title = "Document"
    first = re.search(r'<h1 id="[^"]*"><a[^>]*>#</a>(.*?)</h1>', body, flags=re.S)
    if first:
        title = first.group(1)
        body = body[: first.start()] + body[first.end():]
    return body, title, toc


STYLE = """
:root {
  --bg: #fbfaf7; --surface: #ffffff; --surface-2: #f3f1ec; --text: #1f2328; --muted: #5d6470;
  --border: #e3dfd6; --accent: #3a5ccc; --accent-soft: #e8edfb; --code-bg: #f5f3ee;
  --note: #3a5ccc; --important: #8250df; --warning: #b06a00;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #15171c; --surface: #1c1f26; --surface-2: #232733; --text: #e6e8ec; --muted: #9aa3b2;
    --border: #2f3442; --accent: #8fa8ff; --accent-soft: #232b45; --code-bg: #1f232c;
    --note: #8fa8ff; --important: #c4a5ff; --warning: #e0a44a;
  }
}
:root[data-theme="dark"] {
  --bg: #15171c; --surface: #1c1f26; --surface-2: #232733; --text: #e6e8ec; --muted: #9aa3b2;
  --border: #2f3442; --accent: #8fa8ff; --accent-soft: #232b45; --code-bg: #1f232c;
  --note: #8fa8ff; --important: #c4a5ff; --warning: #e0a44a;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 16px; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif;
}
.layout { display: grid; grid-template-columns: 280px minmax(0, 1fr); max-width: 1320px; margin: 0 auto; }
nav.toc {
  position: sticky; top: 0; align-self: start; height: 100vh; overflow-y: auto;
  padding: 28px 18px 28px 24px; border-right: 1px solid var(--border); font-size: 14px;
}
nav.toc .brand { font-weight: 650; font-size: 15px; margin: 0 0 4px; }
nav.toc .sub { color: var(--muted); font-size: 12.5px; margin: 0 0 18px; }
nav.toc ul { list-style: none; margin: 0; padding: 0; }
nav.toc li { margin: 2px 0; }
nav.toc li.toc-3 { padding-left: 14px; font-size: 13px; }
nav.toc a { color: var(--muted); text-decoration: none; display: block; padding: 3px 8px; border-radius: 6px; }
nav.toc a:hover { color: var(--text); background: var(--surface-2); }
nav.toc a.active { color: var(--accent); background: var(--accent-soft); }
main { padding: 40px 48px 96px; min-width: 0; }
article { max-width: 880px; }
header.doc h1 { font-size: 34px; line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.01em; }
header.doc .meta { color: var(--muted); font-size: 14px; margin: 0 0 28px; }
h2 { font-size: 25px; margin: 56px 0 14px; padding-top: 18px; border-top: 1px solid var(--border); }
h3 { font-size: 20px; margin: 36px 0 10px; }
h4 { font-size: 17px; margin: 28px 0 8px; }
h2, h3, h4 { position: relative; }
.anchor { position: absolute; left: -1.1em; color: var(--border); text-decoration: none; font-weight: 400; opacity: 0; }
h2:hover .anchor, h3:hover .anchor, h4:hover .anchor { opacity: 1; }
p, li { max-width: 76ch; }
a { color: var(--accent); }
code { font: 0.88em/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: var(--code-bg); padding: 0.1em 0.35em; border-radius: 4px; }
pre { background: var(--code-bg); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; overflow-x: auto; }
pre code { background: none; padding: 0; font-size: 13.5px; line-height: 1.55; }
pre:has(code.language-text) { white-space: pre-wrap; border-left: 3px solid var(--accent); }
pre:has(code.language-text) code { white-space: pre-wrap; word-break: break-word; }
figure.diagram { margin: 22px 0; padding: 18px; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow-x: auto; }
figure.diagram pre.mermaid { background: none; border: 0; padding: 0; margin: 0; text-align: center; white-space: pre; }
figure.diagram svg { max-width: 100%; height: auto; }
.table-wrap { overflow-x: auto; margin: 18px 0; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 14.5px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { background: var(--surface-2); font-weight: 600; }
tr:last-child td { border-bottom: 0; }
blockquote { margin: 20px 0; padding: 12px 16px; border-left: 4px solid var(--border); background: var(--surface); border-radius: 0 10px 10px 0; }
blockquote p { margin: 6px 0; }
.alert-title { font-weight: 650; font-size: 14px; text-transform: uppercase; letter-spacing: 0.04em; margin: 0 0 4px !important; }
.alert-note { border-left-color: var(--note); } .alert-note .alert-title { color: var(--note); }
.alert-tip { border-left-color: var(--note); } .alert-tip .alert-title { color: var(--note); }
.alert-important { border-left-color: var(--important); } .alert-important .alert-title { color: var(--important); }
.alert-warning, .alert-caution { border-left-color: var(--warning); }
.alert-warning .alert-title, .alert-caution .alert-title { color: var(--warning); }
details { margin: 16px 0; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); padding: 0 16px; }
details[open] { padding-bottom: 8px; }
summary { cursor: pointer; padding: 10px 0; font-weight: 600; color: var(--accent); }
.theme-toggle { position: fixed; top: 14px; right: 16px; z-index: 5; border: 1px solid var(--border); background: var(--surface); color: var(--muted); border-radius: 8px; padding: 6px 10px; font-size: 13px; cursor: pointer; }
@media (max-width: 960px) {
  .layout { grid-template-columns: 1fr; }
  nav.toc { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--border); padding: 20px 16px; }
  nav.toc ul { columns: 2; column-gap: 16px; }
  main { padding: 24px 16px 72px; }
  .anchor { display: none; }
}
@media (max-width: 560px) {
  nav.toc ul { columns: 1; }
  nav.toc li.toc-3 { display: none; }
  header.doc h1 { font-size: 27px; }
  h2 { font-size: 22px; }
}
"""

SCRIPT = """
(function () {
  var root = document.documentElement;
  var stored = null;
  try { stored = localStorage.getItem("doc-theme"); } catch (e) {}
  if (stored) root.setAttribute("data-theme", stored);
  function dark() {
    var t = root.getAttribute("data-theme");
    return t ? t === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  var sources = [];
  document.querySelectorAll("pre.mermaid").forEach(function (el) { sources.push(el.textContent); });
  function render() {
    if (!window.mermaid) return;
    var els = document.querySelectorAll("pre.mermaid");
    els.forEach(function (el, i) { el.removeAttribute("data-processed"); el.textContent = sources[i]; });
    mermaid.initialize({
      startOnLoad: false, securityLevel: "strict", theme: dark() ? "dark" : "neutral",
      flowchart: { curve: "basis", padding: 12, wrappingWidth: 300 }, er: { fontSize: 13 }
    });
    mermaid.run({ nodes: els });
  }
  render();
  document.getElementById("theme").addEventListener("click", function () {
    var next = dark() ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("doc-theme", next); } catch (e) {}
    render();
  });
  // Highlight the section in view.
  var links = {};
  document.querySelectorAll("nav.toc a").forEach(function (a) { links[a.getAttribute("href").slice(1)] = a; });
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting && links[e.target.id]) {
        Object.values(links).forEach(function (a) { a.classList.remove("active"); });
        links[e.target.id].classList.add("active");
      }
    });
  }, { rootMargin: "0px 0px -75% 0px" });
  document.querySelectorAll("h2[id], h3[id]").forEach(function (h) { observer.observe(h); });
})();
"""


def page(title, source, toc, body):
    items = "\n".join(f'<li class="toc-{level}"><a href="#{ident}">{text}</a></li>' for level, ident, text in toc)
    plain_title = html.escape(re.sub(r"<[^>]+>", "", html.unescape(title)))
    built = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{plain_title}</title>
<style>{STYLE}</style>
</head>
<body>
<button class="theme-toggle" type="button" id="theme">Theme</button>
<div class="layout">
<nav class="toc" aria-label="Contents">
<p class="brand">tabkeeper</p>
<p class="sub">{html.escape(source)}</p>
<ul>
{items}
</ul>
</nav>
<main>
<article>
<header class="doc"><h1>{title}</h1>
<p class="meta">From <code>{html.escape(source)}</code> · built {built}</p></header>
{body}
</article>
</main>
</div>
<script src="mermaid.min.js"></script>
<script>{SCRIPT}</script>
</body>
</html>
"""


def serve(directory, port):
    handler = partial(SimpleHTTPRequestHandler, directory=directory)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Serving http://127.0.0.1:{port}/ (Ctrl-C stops it)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", nargs="?", default="docs/tagging-algorithm.md")
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--serve", nargs="?", type=int, const=8787, metavar="PORT")
    args = parser.parse_args()
    source = os.path.abspath(args.source)
    stem = os.path.splitext(os.path.basename(source))[0]
    default = "tagging-doc" if stem == "tagging-algorithm" else f"{stem}-doc"
    out_dir = os.path.abspath(args.out or os.path.join(ROOT, "target", default))
    os.makedirs(out_dir, exist_ok=True)

    body, title, toc = build_body(markdown_to_html(open(source).read()))
    ensure_mermaid(out_dir)
    out = os.path.join(out_dir, "index.html")
    with open(out, "w") as f:
        f.write(page(title, os.path.relpath(source, ROOT), toc, body))
    diagrams = body.count('<figure class="diagram">')
    print(f"Wrote {os.path.relpath(out, ROOT)}: {diagrams} diagrams, {len(toc)} contents entries")
    if args.serve:
        serve(out_dir, args.serve)


if __name__ == "__main__":
    main()

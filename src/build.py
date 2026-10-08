#!/usr/bin/env python
"""Static site generator for the GraphTok GitHub Pages site.

Usage: python build.py <repo_clone_dir> <output_dir>

Renders the repository markdown into docs/*.html with a shared shell, copies
the hand-written homepage and stylesheet, and emits pygments CSS.
"""
import os
import re
import shutil
import sys
import posixpath

import markdown
from pygments.formatters import HtmlFormatter

SRC = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/Msiavashi/GraphTok"
BLOB = REPO_URL + "/blob/main/"

# (repo path, output page, nav title)
PAGES = [
    ("README.md", "index.html", "Overview"),
    ("docs/BUILD.md", "build.html", "Building"),
    ("docs/ARCHITECTURE.md", "architecture.html", "Architecture"),
    ("docs/ADDING_A_TOKENIZER.md", "adding-a-tokenizer.html", "Adding a tokenizer"),
    ("docs/DEVELOPMENT.md", "development.html", "Development"),
    ("integrations/dynamo/README.md", "dynamo.html", "NVIDIA Dynamo backend"),
    ("CONTRIBUTING.md", "contributing.html", "Contributing"),
    ("CHANGELOG.md", "changelog.html", "Changelog"),
    ("CODE_OF_CONDUCT.md", "code-of-conduct.html", "Code of conduct"),
    ("SECURITY.md", "security.html", "Security policy"),
]
PAGE_FOR = {src: out for src, out, _ in PAGES}

SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - GraphTok</title>
<meta name="description" content="GraphTok documentation: {title}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="{root}style.css">
<link rel="stylesheet" href="{root}pygments.css">
<link rel="icon" href="{root}favicon.svg" type="image/svg+xml">
</head>
<body class="docs">
<header class="top">
  <div class="wrap">
    <a class="brand" href="{root}index.html"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="5" cy="12" r="2.2"/><circle cx="12" cy="5" r="2.2"/><circle cx="12" cy="19" r="2.2"/><circle cx="19" cy="12" r="2.2"/><path d="M7 11l3.3-4.4M7 13l3.3 4.4M14 6.6L17 10M14 17.4L17 14"/></svg>GraphTok</a>
    <nav>
      <a href="{root}index.html">Home</a>
      <a href="{root}docs/index.html" class="active">Docs</a>
      <a href="{repo}">GitHub</a>
    </nav>
  </div>
</header>
<div class="wrap docs-layout">
  <aside class="sidebar">
    <p class="side-title">Documentation</p>
    <ul>
{sidebar}
    </ul>
  </aside>
  <main class="content">
{body}
  </main>
</div>
<footer class="foot">
  <div class="wrap">
    <span>GraphTok is released under the Apache License 2.0.</span>
    <span><a href="{repo}">GitHub</a> &middot; <a href="{repo}/blob/main/LICENSE">License</a> &middot; <a href="{root}docs/security.html">Security</a></span>
  </div>
</footer>
</body>
</html>
"""


def rewrite_link(href, src_path):
    """Map intra-repo markdown links to site pages or GitHub blob URLs."""
    if re.match(r"^(https?:|mailto:|#)", href):
        return href
    target, _, frag = href.partition("#")
    base = posixpath.dirname(src_path)
    norm = posixpath.normpath(posixpath.join(base, target)) if target else src_path
    frag = ("#" + frag) if frag else ""
    if norm in PAGE_FOR:
        return PAGE_FOR[norm] + frag
    return BLOB + norm + frag


def preprocess(text, src_path):
    if src_path == "README.md":
        # Drop the shields.io badge block.
        text = re.sub(r"^\[!\[[^\n]*\n", "", text, flags=re.M)
    def sub(m):
        return "[" + m.group(1) + "](" + rewrite_link(m.group(2), src_path) + ")"
    # [text](link) not preceded by '!' (images) and not inside code spans is
    # good enough for these documents.
    return re.sub(r"(?<!\!)\[([^\]]*)\]\(([^)\s]+)\)", sub, text)


def render(repo, out):
    docs_out = os.path.join(out, "docs")
    os.makedirs(docs_out, exist_ok=True)
    sidebar = "\n".join(
        '      <li><a href="{o}" data-page="{o}">{t}</a></li>'.format(o=o, t=t)
        for _, o, t in PAGES
    )
    problems = []
    for src, outname, title in PAGES:
        path = os.path.join(repo, src)
        with open(path, encoding="utf-8") as f:
            text = preprocess(f.read(), src)
        md = markdown.Markdown(
            extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"],
            extension_configs={
                "codehilite": {"guess_lang": False, "css_class": "highlight"},
                "toc": {"permalink": False},
            },
        )
        body = md.convert(text)
        if "&lt;" in body and "<pre" not in body and "<code" not in body:
            problems.append(src)
        side = sidebar.replace(
            'href="{0}" data-page="{0}"'.format(outname),
            'href="{0}" class="active"'.format(outname),
        ).replace(' data-page="', ' data-x="')
        side = re.sub(r' data-x="[^"]*"', "", side)
        page = SHELL.format(title=title, root="../", repo=REPO_URL, sidebar=side, body=body)
        with open(os.path.join(docs_out, outname), "w", encoding="utf-8") as f:
            f.write(page)
    return problems


def pygments_css(out):
    light = HtmlFormatter(style="default").get_style_defs(".highlight")
    dark = HtmlFormatter(style="github-dark").get_style_defs(".highlight")
    css = (
        light
        + "\n@media (prefers-color-scheme: dark) {\n:root:not([data-theme=\"light\"]) "
        + dark.replace("\n", "\n:root:not([data-theme=\"light\"]) ")
        + "\n}\n"
        + ".highlight { background: var(--code-bg) !important; }\n"
    )
    with open(os.path.join(out, "pygments.css"), "w") as f:
        f.write(css)


def main():
    repo, out = sys.argv[1], sys.argv[2]
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for name in ("index.html", "style.css", "favicon.svg"):
        shutil.copy(os.path.join(SRC, name), os.path.join(out, name))
    open(os.path.join(out, ".nojekyll"), "w").close()
    pygments_css(out)
    problems = render(repo, out)
    pages = sorted(
        os.path.relpath(os.path.join(d, f), out)
        for d, _, fs in os.walk(out) for f in fs
    )
    print("pages:", *pages, sep="\n  ")
    print("markdown that may not have converted cleanly:", problems or "none")


if __name__ == "__main__":
    main()

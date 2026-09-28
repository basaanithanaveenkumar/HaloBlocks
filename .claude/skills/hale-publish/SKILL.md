---
name: hale-publish
description: Update and rebuild this repository's public-facing material — the arXiv paper (paper/), the GitHub Pages project page (project-page/), the docs (docs/), the Mermaid architecture diagrams and the blog (docs/blog/). Use after changing the architecture or results, or when asked to write up, publish, or refresh the paper, page, docs or blog.
---

# Publishing: paper, project page, docs, blog

| Material | Location | Source of truth |
|---|---|---|
| arXiv paper | `paper/main.tex`, `paper/references.bib` | code + recorded experiments |
| Project page | `project-page/index.html` | mirrors the paper abstract, diagrams and BibTeX |
| Docs | `docs/*.md` | code |
| Architecture diagrams | `docs/architecture.md` (Mermaid) | code |
| Blog | `docs/blog/*.md` | narrative version of the paper |

## Rules

- **Never invent numbers.** Tables in the paper hold only results you actually ran and
  can reproduce with a command in the repo. Unrun experiments stay marked `TBD` with a
  `% TODO` comment.
- When the architecture changes, update in this order: code → `docs/architecture.md`
  → paper method section and figure → project page diagram → blog if the story changed.
- Keep the Mermaid source identical between `docs/architecture.md` and
  `project-page/index.html` (the page renders the same diagrams with mermaid@11).
- Mermaid labels containing brackets, parentheses or `<tokens>` must be quoted:
  `A["&lt;image&gt; tokens (196)"]`.

## Build the paper

```bash
cd paper && latexmk -pdf main.tex      # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

For arXiv submission upload `main.tex`, `main.bbl` and `references.bib` (arXiv does not
run BibTeX reliably, so include the `.bbl`). `latexmk -c` cleans aux files.

## Project page

`project-page/` is static HTML (no build step). Preview with
`python -m http.server -d project-page 8000`.

`.github/workflows/pages.yml` compiles the paper and publishes `project-page/` plus
`paper.pdf` to GitHub Pages on every push to `main` that touches `paper/` or
`project-page/`. One-time setup: *Settings → Pages → Source: GitHub Actions*.
The page is then served at `https://basaanithanaveenkumar.github.io/HaloBlocks/`.

## Blog

One Markdown file per post in `docs/blog/`, named `YYYY-MM-DD-slug.md`, with a title,
date and a one-paragraph summary at the top. Link new posts from `docs/blog/README.md`.

# Wiki content (source of truth)

The Markdown files in this directory are the **canonical source** of the
GitHub Wiki at https://github.com/bettyguo/realm-retrieve/wiki. Edit them
here, send a PR, get a review, and CI mirrors the merged content to the
wiki automatically (or run [`scripts/sync_wiki.sh`](../../scripts/sync_wiki.sh)
locally for a one-shot push).

## Layout

| File              | Wiki URL                                                  |
|-------------------|-----------------------------------------------------------|
| `Home.md`         | https://github.com/bettyguo/realm-retrieve/wiki           |
| `Architecture.md` | https://github.com/bettyguo/realm-retrieve/wiki/Architecture |
| `Reproducibility.md` | https://github.com/bettyguo/realm-retrieve/wiki/Reproducibility |
| `FAQ.md`          | https://github.com/bettyguo/realm-retrieve/wiki/FAQ        |
| `Glossary.md`     | https://github.com/bettyguo/realm-retrieve/wiki/Glossary   |
| `Roadmap.md`      | https://github.com/bettyguo/realm-retrieve/wiki/Roadmap    |
| `Citation.md`     | https://github.com/bettyguo/realm-retrieve/wiki/Citation   |
| `_Sidebar.md`     | (renders as the wiki's sidebar — every page)               |
| `_Footer.md`      | (renders as the wiki's footer — every page)                |

## Conventions

- **Filename = URL slug.** GitHub Wikis URL-encode the filename verbatim.
  Use `PascalCase` rather than spaces; we don't link with hyphens.
- **Relative wiki links** use `[Architecture](Architecture)` (no extension,
  no leading slash). On the rendered wiki, these resolve via GitHub's wiki
  router; here, they render as broken links — that's expected.
- **Cross-links to the repo** use absolute `https://github.com/bettyguo/…`
  URLs since the wiki's URL space is disjoint from the repo's URL space.

# Discussion drafts

The Markdown files here are the **canonical source** of the welcome /
landing posts we want to seed in each Discussion category once the
**Discussions** feature is enabled on https://github.com/bettyguo/realm-retrieve.

Until the feature is on, this directory is editorial scaffolding. Once
it's on, run:

```bash
scripts/seed_discussions.py           # pushes every .md as a new discussion
```

## File → category mapping

| File                                | Category       | Why this is the first post                                  |
|-------------------------------------|----------------|-------------------------------------------------------------|
| `announcements/01-v1.0.0.md`        | Announcements  | The launch post (mirrors the v1.0.0 Release notes).         |
| `general/01-welcome.md`             | General        | "Welcome — read this first."                                |
| `ideas/01-whats-next.md`            | Ideas          | A solicit-the-community thread for v1.2 / v2.0 priorities.  |
| `q-and-a/01-troubleshooting.md`     | Q&A            | Common errors + how to file a good bug report.              |
| `show-and-tell/01-template.md`      | Show and tell  | "Built something on ReaLM-Retrieve? Drop it here."          |

## Front-matter schema

```yaml
---
title:    "Short, descriptive title"
category: announcements | general | ideas | q-and-a | show-and-tell | polls
pin:      true | false             # whether to pin the post in its category
lock:     true | false             # whether to lock the thread (e.g. announcements)
labels:   [optional]               # custom labels — created on demand
---
```

The body that follows is plain GitHub-flavoured Markdown.

## Conventions

- One post per category at seed time; subsequent posts come from the community.
- Announcement posts are **locked** (read-only). Discussion happens in *General*.
- Q&A posts use a leading question; the resolved answer is the marked
  "Answer" once we have one.
- All posts cross-link to the wiki / issues / docs to teach the URL space.

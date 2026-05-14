# Issue drafts

This directory contains the **source of truth** for the project's GitHub issues.
Each file is a self-contained issue with a YAML frontmatter block describing
its labels, milestone, and lifecycle state.

```
.github/ISSUE_DRAFTS/
├── open/        # issues that should still be open on GitHub
└── closed/      # issues that ship already-resolved (referenced from CHANGELOG)
```

## Pushing to GitHub

Set up the remote first (`gh repo create` or `git remote add origin …`), then:

```bash
bash  scripts/seed_issues.sh          # macOS / Linux / WSL / Git-Bash
pwsh  scripts/seed_issues.ps1         # Windows PowerShell
```

The script reads every `*.md` file, creates the milestones if needed, opens
issues for the `open/` set, and opens-then-closes issues from `closed/` with a
"Closed by commit X" reference. Re-running is idempotent — it skips titles that
already exist on the remote.

## Frontmatter schema

```yaml
title:    "Short, imperative issue title"
labels:   ["bug", "good first issue", ...]
milestone: "v1.1"                       # optional, created on demand
state:    open                          # open | closed
assignees: ["bettyguo"]                 # optional
opened:   "2026-04-22"                  # ISO date, used for the seed script
closed:   "2026-04-22"                  # ISO date, only for state=closed
closed_by: "fix(evaluate): correct Hydra config_path (#1)"
```

## Conventions

- Use Conventional-Commit-style titles (`fix:`, `feat:`, `docs:`, `chore:`…).
- The body is plain Markdown — fenced code blocks, tables, checklists welcome.
- `good first issue` should be **small enough to finish in a weekend** and
  include a "Hints" section with file paths and starter snippets.

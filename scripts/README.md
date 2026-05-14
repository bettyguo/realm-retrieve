# Maintenance scripts

| Script | Purpose | Platform |
|--------|---------|----------|
| [`seed_issues.sh`](seed_issues.sh) | Push the issue drafts under [`.github/ISSUE_DRAFTS/`](../.github/ISSUE_DRAFTS/) to the GitHub remote using `gh` CLI. Idempotent. | macOS / Linux / WSL / Git-Bash |
| [`seed_issues.ps1`](seed_issues.ps1) | The PowerShell equivalent of the above. | Windows |

## First-time setup

```bash
# 1. Authenticate
gh auth login

# 2. Push the local repo somewhere with issues enabled (e.g. github.com/bettyguo)
gh repo create bettyguo/realm-retrieve --public --source=. --remote=origin --push

# 3. Seed
bash scripts/seed_issues.sh
```

The script:

1. Parses YAML frontmatter from every `*.md` under `.github/ISSUE_DRAFTS/`.
2. Creates each milestone on demand (`v1.0`, `v1.1 (2026-06-11)`, …).
3. Opens each issue (skipping titles that already exist on the remote, so
   re-runs are safe).
4. Closes anything with `state: closed`, attaching a "Closed by *<commit>*"
   comment so the history is auditable.

## Refreshing one issue

Edit the markdown file, then re-run the script with a path filter:

```bash
bash scripts/seed_issues.sh .github/ISSUE_DRAFTS/open/06-modernize-type-hints.md
```

If the title is unchanged, the script skips the create step — to apply
edits to an existing issue, delete it on GitHub first or use
`gh issue edit <number>` manually.

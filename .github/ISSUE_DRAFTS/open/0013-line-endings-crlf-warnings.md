---
title:    "chore(repo): add .gitattributes to pin LF line-endings (silence CRLF warnings)"
labels:   ["good first issue", "tech-debt", "dx"]
milestone: "v1.1"
state:    open
opened:   "2026-05-14"
---

## Symptom

Every `git commit` from a Windows worktree prints:

```
warning: in the working copy of 'scripts/seed_issues.py', LF will be
replaced by CRLF the next time Git touches it
warning: in the working copy of '.github/workflows/sync-wiki.yml', LF will be ...
...
```

Repeated for every text file. No functional impact — just noisy and
distracts from real warnings.

## Fix

Add `.gitattributes` at the repo root:

```
# Pin every text file to LF in the index. Shell scripts must stay LF or
# they'll fail with `bad interpreter: /bin/bash^M` on Linux runners.

* text=auto eol=lf

*.sh   text eol=lf
*.py   text eol=lf
*.yml  text eol=lf
*.yaml text eol=lf
*.md   text eol=lf
*.toml text eol=lf
*.cfg  text eol=lf
*.json text eol=lf

# Binary
*.png binary
*.jpg binary
*.svg text eol=lf
*.pdf binary
*.bin binary
*.pt  binary
```

Then run once on the maintainer's machine:

```bash
git add --renormalize .
git commit -m "chore(repo): normalize line endings to LF"
```

## Acceptance

- [ ] `.gitattributes` lives at repo root.
- [ ] A fresh commit on Windows produces zero `CRLF` warnings.
- [ ] CI on Linux still passes (no `bad interpreter` errors).

## Hints

- This is a 10-minute task, perfect for a first PR.
- Don't forget `scripts/sync_wiki.sh` and `scripts/seed_issues.sh` — they
  *must* be LF or the GitHub Actions runner can't execute them.

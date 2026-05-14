#!/usr/bin/env bash
# =============================================================================
# Mirror docs/wiki/*.md to the GitHub Wiki repository.
#
# GitHub Wikis are stored in a separate git repository: <repo>.wiki.git.
# This script clones it, replaces its contents with docs/wiki/, and pushes.
#
# Run locally:    scripts/sync_wiki.sh
# Run from CI:    .github/workflows/sync-wiki.yml does this automatically
#                  on every push to main that touches docs/wiki/.
#
# Requirements:
#   - git authenticated to push to <repo>.wiki.git (the main repo's PAT works,
#     provided it has `Contents: write` on the same repo).
#   - The wiki must be **enabled** in repo settings (Settings → Features → Wikis).
#     If you see "Repository not found" it usually means the wiki is disabled.
# =============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/bettyguo/realm-retrieve.wiki.git}"
SRC_DIR="${SRC_DIR:-$(git rev-parse --show-toplevel)/docs/wiki}"
TMP_DIR="$(mktemp -d -t realm-wiki-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ ! -d "$SRC_DIR" ]]; then
  echo "❌ Source directory not found: $SRC_DIR" >&2
  exit 1
fi

# If GH_TOKEN is set, embed it in the clone URL so push works in CI.
CLONE_URL="$REPO_URL"
if [[ -n "${GH_TOKEN:-}" ]]; then
  CLONE_URL="${REPO_URL/https:\/\//https://x-access-token:$GH_TOKEN@}"
fi

echo "→ cloning wiki: $REPO_URL"
if ! git clone --depth=1 "$CLONE_URL" "$TMP_DIR/wiki" 2>/dev/null; then
  cat >&2 <<'EOM'
❌ Could not clone the wiki repo.

Possible causes:
  1. The Wiki feature is disabled. Enable it at:
     Settings → Features → Wikis (needs Administration permission).
  2. The token doesn't have Contents:write on the repo.
  3. The wiki has never been initialised — create any page once in the
     browser, then re-run this script.
EOM
  exit 1
fi

echo "→ syncing $SRC_DIR/*  →  wiki"
# Wipe everything tracked by the wiki repo (keep .git).
find "$TMP_DIR/wiki" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp -r "$SRC_DIR"/* "$TMP_DIR/wiki/"

# README.md is editorial scaffolding; the wiki landing page is Home.md.
rm -f "$TMP_DIR/wiki/README.md"

cd "$TMP_DIR/wiki"
git add -A
if git diff --cached --quiet; then
  echo "✓ wiki already up to date — no commit."
  exit 0
fi

git -c user.name="realm-retrieve-bot" \
    -c user.email="bot@realm-retrieve.dev" \
    commit -m "docs(wiki): sync from docs/wiki/ on $(date -u +%Y-%m-%dT%H:%M:%SZ)"

git push origin HEAD:master
echo "✓ wiki pushed."

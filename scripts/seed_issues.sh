#!/usr/bin/env bash
# =============================================================================
# seed_issues.sh — push the issue drafts under .github/ISSUE_DRAFTS/ to GitHub
#                  using the gh CLI.
#
# Usage:
#   bash scripts/seed_issues.sh                # all drafts
#   bash scripts/seed_issues.sh open/06-*.md   # single draft
#
# Re-runs are idempotent: titles that already exist on the remote are skipped.
# Requires gh (>= 2.30) and a remote configured for this repo.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRAFTS_DIR="$ROOT/.github/ISSUE_DRAFTS"

# ---------- prerequisites ----------------------------------------------------
command -v gh   >/dev/null || { echo "gh CLI not found"; exit 1; }
command -v awk  >/dev/null || { echo "awk not found";     exit 1; }
gh auth status  >/dev/null 2>&1 || { echo "Run 'gh auth login' first"; exit 1; }

# ---------- helpers ----------------------------------------------------------
front_value() {        # front_value <file> <key>
    awk -v K="$2" '
        /^---$/        { c++; next }
        c==1 && $1 == K":" { sub("^[^:]+:[[:space:]]*", ""); gsub(/^"|"$/, ""); print; exit }
    ' "$1"
}

front_list() {         # YAML inline list: labels: ["a", "b"]
    awk -v K="$2" '
        /^---$/ { c++; next }
        c==1 && $1 == K":" {
            sub("^[^:]+:[[:space:]]*", "")
            gsub(/^\[|\]$/, ""); gsub("\"", "")
            print; exit
        }
    ' "$1"
}

body_of() {            # body_of <file>
    awk '/^---$/ { c++; if (c == 2) { getline; capture=1 } next }
         capture' "$1"
}

ensure_milestone() {
    local title="$1"
    [[ -z "$title" ]] && return
    if ! gh api "repos/{owner}/{repo}/milestones" --paginate \
        | jq -e --arg t "$title" '.[] | select(.title == $t)' >/dev/null; then
        echo "  ↳ creating milestone $title"
        gh api -X POST "repos/{owner}/{repo}/milestones" -f title="$title" >/dev/null
    fi
}

issue_exists() {
    gh issue list --state all --search "in:title \"$1\"" --json title \
        | jq -e --arg t "$1" '.[] | select(.title == $t)' >/dev/null
}

# ---------- main -------------------------------------------------------------
shopt -s nullglob
FILES=("$@")
if [[ ${#FILES[@]} -eq 0 ]]; then
    FILES=("$DRAFTS_DIR"/closed/*.md "$DRAFTS_DIR"/open/*.md)
fi

for f in "${FILES[@]}"; do
    [[ -f "$f" ]] || continue

    title=$(front_value "$f" title)
    state=$(front_value "$f" state)
    milestone=$(front_value "$f" milestone)
    closed_by=$(front_value "$f" closed_by)
    labels_csv=$(front_list "$f" labels | tr ',' '\n' | sed 's/^ *//;s/ *$//' | paste -sd, -)
    body=$(body_of "$f")

    echo "→ $title"
    if issue_exists "$title"; then
        echo "  ↳ already exists on remote, skipping"
        continue
    fi

    ensure_milestone "$milestone"

    args=(--title "$title" --body "$body")
    [[ -n "$labels_csv" ]] && args+=(--label "$labels_csv")
    [[ -n "$milestone"  ]] && args+=(--milestone "$milestone")

    url=$(gh issue create "${args[@]}")
    echo "  ↳ created $url"

    if [[ "$state" == "closed" ]]; then
        comment="Closed by **$closed_by** — shipped in v1.0."
        gh issue close "$url" --comment "$comment" >/dev/null
        echo "  ↳ closed (${closed_by})"
    fi
done

echo "✔ done"

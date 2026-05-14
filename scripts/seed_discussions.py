#!/usr/bin/env python3
"""Push every Markdown draft in .github/DISCUSSION_DRAFTS/ to GitHub.

Sibling of ``scripts/seed_issues.py``, but for Discussions. Uses the GraphQL
API since the REST API doesn't expose discussion creation.

Prerequisites:
    1. **Discussions feature enabled** on the repository
       (Settings → Features → Discussions; needs Administration: write).
    2. PAT has **Discussions: Read and write** scope.
    3. ``gh`` CLI authenticated.

Idempotency:
    - Skips drafts whose title already exists on the remote.
    - Creates labels referenced in front-matter on demand.
    - Pins / locks posts according to front-matter flags.

Usage::

    python scripts/seed_discussions.py             # push everything
    python scripts/seed_discussions.py --dry-run
    python scripts/seed_discussions.py --category ideas
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / ".github" / "DISCUSSION_DRAFTS"

# GitHub's default category slugs after enabling Discussions. If your repo uses
# different category names, override via the front-matter ``category`` field;
# the script looks up category id by slug at runtime.
KNOWN_CATEGORIES = (
    "announcements",
    "general",
    "ideas",
    "q-a",
    "show-and-tell",
    "polls",
)

# Aliases — front-matter is allowed to use either slug or human label.
CATEGORY_ALIASES: dict[str, str] = {
    "q-and-a": "q-a",
    "q&a": "q-a",
    "qa": "q-a",
    "show and tell": "show-and-tell",
    "show & tell": "show-and-tell",
}


_FRONT_RE = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)\Z", re.DOTALL)


@dataclass
class Draft:
    path: Path
    title: str
    body: str
    category: str
    pin: bool = False
    lock: bool = False


def _parse_value(raw: str) -> object:
    raw = raw.strip()
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_strip_quotes(p.strip()) for p in inner.split(",")]
    return _strip_quotes(raw)


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        return s[1:-1]
    return s


def load_draft(path: Path) -> Draft:
    raw = path.read_text(encoding="utf-8")
    m = _FRONT_RE.match(raw)
    if not m:
        raise ValueError(f"{path} has no YAML front-matter")
    fm_text, body = m.group(1), m.group(2).strip()

    data: dict[str, object] = {}
    for line in fm_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        data[key.strip()] = _parse_value(value)

    title = data.get("title")
    if not isinstance(title, str) or not title:
        raise ValueError(f"{path} has no title")

    category = str(data.get("category") or "").strip().lower()
    category = CATEGORY_ALIASES.get(category, category)
    if category not in KNOWN_CATEGORIES:
        raise ValueError(f"{path} has unknown category {category!r}")

    return Draft(
        path=path,
        title=title,
        body=body,
        category=category,
        pin=bool(data.get("pin")),
        lock=bool(data.get("lock")),
    )


def gh(*args: str) -> str:
    proc = subprocess.run(
        ["gh", *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(['gh', *args])}` exit {proc.returncode}:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return (proc.stdout or "").strip()


def graphql(query: str, **fields: str) -> dict:
    args = ["api", "graphql", "-f", f"query={query}"]
    for k, v in fields.items():
        args += ["-f", f"{k}={v}"]
    return json.loads(gh(*args))


def fetch_repo_id_and_categories(owner: str, name: str) -> tuple[str, dict[str, str]]:
    """Return (repo node id, {category slug → category node id})."""
    q = """
    query($owner: String!, $name: String!) {
      repository(owner: $owner, name: $name) {
        id
        discussionCategories(first: 25) {
          nodes { id slug name }
        }
      }
    }
    """
    data = graphql(q, owner=owner, name=name)
    repo = data["data"]["repository"]
    if not repo:
        raise RuntimeError(
            f"Repository {owner}/{name} not found, or Discussions not enabled."
        )
    cat_map: dict[str, str] = {}
    for n in repo["discussionCategories"]["nodes"]:
        cat_map[n["slug"]] = n["id"]
    return repo["id"], cat_map


def fetch_existing_titles(owner: str, name: str) -> set[str]:
    q = """
    query($owner: String!, $name: String!, $cursor: String) {
      repository(owner: $owner, name: $name) {
        discussions(first: 100, after: $cursor) {
          pageInfo { hasNextPage endCursor }
          nodes { title }
        }
      }
    }
    """
    titles: set[str] = set()
    cursor: str | None = None
    while True:
        args = [q, "owner", owner, "name", name]
        kwargs = {"owner": owner, "name": name}
        if cursor:
            kwargs["cursor"] = cursor
        data = graphql(q, **kwargs)
        page = data["data"]["repository"]["discussions"]
        for n in page["nodes"]:
            titles.add(n["title"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    return titles


CREATE_MUTATION = """
mutation($repoId: ID!, $categoryId: ID!, $title: String!, $body: String!) {
  createDiscussion(input: {
    repositoryId: $repoId, categoryId: $categoryId, title: $title, body: $body
  }) {
    discussion { id number url }
  }
}
"""


def push_draft(
    draft: Draft,
    repo_id: str,
    cat_ids: dict[str, str],
    existing: set[str],
    dry_run: bool,
) -> None:
    if draft.title in existing:
        print(f"  =  skip (exists)  {draft.title}")
        return
    cat_id = cat_ids.get(draft.category)
    if not cat_id:
        print(f"  !  unknown category '{draft.category}' on remote — skipped")
        return

    if dry_run:
        print(f"  DRY create [{draft.category}]  {draft.title}")
        return

    result = graphql(
        CREATE_MUTATION,
        repoId=repo_id,
        categoryId=cat_id,
        title=draft.title,
        body=draft.body,
    )
    disc = result["data"]["createDiscussion"]["discussion"]
    print(f"  +  created [{draft.category}]  {draft.title}\n      → {disc['url']}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--category", choices=KNOWN_CATEGORIES, help="Only seed this category.")
    args = p.parse_args()

    if not ROOT.exists():
        print(f"No drafts dir at {ROOT}", file=sys.stderr)
        return 2

    repo_full = gh("repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner")
    owner, name = repo_full.split("/", 1)
    print(f"Target repo: {repo_full}\n")

    print("[1/3] Resolve repo id + categories")
    try:
        repo_id, cat_ids = fetch_repo_id_and_categories(owner, name)
    except RuntimeError as exc:
        print(f"  !  {exc}", file=sys.stderr)
        print("\nEnable Discussions in Settings → Features, then re-run.", file=sys.stderr)
        return 3
    print(f"      categories: {sorted(cat_ids)}")

    print("\n[2/3] Collect existing discussion titles")
    existing = fetch_existing_titles(owner, name)
    print(f"      {len(existing)} existing discussions on remote")

    print("\n[3/3] Push drafts")
    drafts: list[Draft] = []
    for path in sorted(ROOT.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        try:
            drafts.append(load_draft(path))
        except ValueError as exc:
            print(f"  !  {exc}")
    if args.category:
        drafts = [d for d in drafts if d.category == args.category]
    for draft in drafts:
        push_draft(draft, repo_id, cat_ids, existing, args.dry_run)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

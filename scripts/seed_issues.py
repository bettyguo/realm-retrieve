#!/usr/bin/env python3
"""Push every Markdown draft in .github/ISSUE_DRAFTS/ to GitHub.

Pure-stdlib alternative to ``scripts/seed_issues.sh`` — uses only the ``gh``
CLI (no ``yq`` dependency). Idempotent: skips drafts whose title already
exists on the remote, creates the four milestones on demand, normalises
labels, and closes drafts marked ``state: closed`` with a "closed by …"
comment.

Usage::

    python scripts/seed_issues.py             # push everything
    python scripts/seed_issues.py --dry-run   # show what would happen
    python scripts/seed_issues.py --subset open
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / ".github" / "ISSUE_DRAFTS"

MILESTONES: dict[str, tuple[str, str | None]] = {
    "v1.0":      ("Camera-ready bugfix flight (already shipped)",     "2026-05-14"),
    "v1.1":      ("Polish & onboarding (CPU smoke tests, type hints)", "2026-06-11"),
    "v1.2":      ("Plugin architecture & performance",                 "2026-07-09"),
    "v2.0":      ("Research extensions (multilingual, live demo)",     "2026-09-03"),
    "tech-debt": ("Always-on backlog",                                 None),
}

LABEL_COLORS: dict[str, str] = {
    "bug": "d73a4a",
    "enhancement": "a2eeef",
    "good first issue": "7057ff",
    "help wanted": "008672",
    "needs-triage": "ededed",
    "research": "5319e7",
    "ci": "bfd4f2",
    "config": "0e8a16",
    "imports": "fef2c0",
    "training": "c2e0c6",
    "rl": "1d76db",
    "reproducibility": "0e8a16",
    "demo": "ff9f1c",
    "architecture": "5319e7",
    "testing": "bfd4f2",
    "tech-debt": "cccccc",
    "rsus": "d4c5f9",
    "dx": "fbca04",
    "dependencies": "0366d6",
}


# --------------------------------------------------------------------------- #
# YAML front-matter parser (deliberately tiny: drafts use a fixed shape)
# --------------------------------------------------------------------------- #

_FRONT_RE = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)\Z", re.DOTALL)


def _parse_value(raw: str) -> object:
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [
            _strip_quotes(item.strip())
            for item in re.split(r",(?![^\[]*\])", inner)
        ]
    return _strip_quotes(raw)


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        return s[1:-1]
    return s


@dataclass
class Draft:
    path: Path
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    assignees: list[str] = field(default_factory=list)
    milestone: str = ""
    state: str = "open"
    closed_by: str = ""
    closed: str = ""


def load_draft(path: Path) -> Draft:
    raw = path.read_text(encoding="utf-8")
    m = _FRONT_RE.match(raw)
    if not m:
        raise ValueError(f"{path} has no YAML front-matter")
    fm_text, body = m.group(1), m.group(2).strip()

    data: dict[str, object] = {}
    for line in fm_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        data[key.strip()] = _parse_value(value)

    title = data.get("title")
    if not isinstance(title, str) or not title:
        raise ValueError(f"{path} has no title")

    labels    = [str(v) for v in (data.get("labels")    or []) if v]
    assignees = [str(v) for v in (data.get("assignees") or []) if v]

    return Draft(
        path=path,
        title=title,
        body=body,
        labels=labels,
        assignees=assignees,
        milestone=str(data.get("milestone") or ""),
        state=str(data.get("state") or "open"),
        closed_by=str(data.get("closed_by") or ""),
        closed=str(data.get("closed") or ""),
    )


# --------------------------------------------------------------------------- #
# gh shell-out
# --------------------------------------------------------------------------- #


def gh(*args: str, check: bool = True, capture: bool = True, input_text: str | None = None) -> str:
    cmd = ["gh", *args]
    proc = subprocess.run(
        cmd,
        input=input_text,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(cmd)}` failed (exit {proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return (proc.stdout or "").strip()


def gh_json(*args: str) -> object:
    return json.loads(gh(*args) or "null")


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #


def ensure_milestones(repo: str, dry_run: bool) -> None:
    existing = gh_json("api", f"repos/{repo}/milestones?state=all&per_page=100")
    titles = {m["title"] for m in existing}  # type: ignore[index]
    for name, (desc, due) in MILESTONES.items():
        if name in titles:
            print(f"  =  milestone exists   {name}")
            continue
        print(f"  +  milestone created  {name}")
        if dry_run:
            continue
        args = ["api", "--method", "POST", f"repos/{repo}/milestones",
                "-f", f"title={name}", "-f", f"description={desc}"]
        if due:
            args += ["-f", f"due_on={due}T23:59:59Z"]
        gh(*args)


def ensure_labels(dry_run: bool) -> None:
    existing = gh_json("label", "list", "--limit", "200", "--json", "name")
    have = {item["name"] for item in existing}  # type: ignore[index]
    for name, color in LABEL_COLORS.items():
        if name in have:
            continue
        print(f"  +  label created      {name}")
        if dry_run:
            continue
        gh("label", "create", name, "--color", color, "--force")


def existing_issues() -> dict[str, dict]:
    """Map title → {number, state, labels, milestone} for every issue on remote."""
    data = gh_json(
        "issue", "list",
        "--state", "all",
        "--limit", "500",
        "--json", "number,title,state,labels,milestone",
    )
    return {item["title"]: item for item in data}  # type: ignore[index]


def push_draft(draft: Draft, taken: dict[str, dict], dry_run: bool) -> None:
    # If an issue with this title already exists, don't create a duplicate.
    # But if the draft says `state: closed` and the remote one is open, run the
    # close ritual (comment + close) so the remote matches our source of truth.
    if draft.title in taken:
        remote = taken[draft.title]
        num = str(remote["number"])
        remote_state = (remote.get("state") or "").lower()
        if draft.state == "closed" and remote_state != "closed":
            print(f"  ~  existing #{num} OPEN → closing per draft  {draft.title}")
            if dry_run:
                return
            notes: list[str] = []
            if draft.closed_by:
                notes.append(f"Closed by `{draft.closed_by}`.")
            if draft.closed:
                notes.append(f"Closed on {draft.closed}.")
            if notes:
                gh("issue", "comment", num, "--body", "\n".join(notes))
            gh("issue", "close", num)
            print(f"      ↳ closed #{num}")
            return
        print(f"  =  skip (exists)      {draft.title}  (#{num}, {remote_state})")
        return

    args = ["issue", "create", "--title", draft.title, "--body", draft.body]
    for lbl in draft.labels:
        args += ["--label", lbl]
    for a in draft.assignees:
        args += ["--assignee", a]
    if draft.milestone:
        args += ["--milestone", draft.milestone]

    if dry_run:
        print(f"  DRY create           {draft.title}")
        return

    url = gh(*args)
    print(f"  +  created            {draft.title}\n      → {url}")
    taken[draft.title] = {"number": int(url.rsplit("/", 1)[-1]), "state": "OPEN"}

    if draft.state == "closed":
        num = url.rsplit("/", 1)[-1]
        notes: list[str] = []
        if draft.closed_by:
            notes.append(f"Closed by `{draft.closed_by}`.")
        if draft.closed:
            notes.append(f"Closed on {draft.closed}.")
        if notes:
            gh("issue", "comment", num, "--body", "\n".join(notes))
        gh("issue", "close", num)
        print(f"      ↳ closed")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen.")
    parser.add_argument("--subset", choices=["open", "closed"], help="Only this subset.")
    args = parser.parse_args()

    if not ROOT.exists():
        print(f"No issue drafts dir at {ROOT}", file=sys.stderr)
        return 2

    repo = gh("repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner")
    print(f"Target repo: {repo}")

    print("\n[1/4] Ensure milestones")
    ensure_milestones(repo, args.dry_run)

    print("\n[2/4] Ensure labels")
    ensure_labels(args.dry_run)

    print("\n[3/4] Collect existing issue titles")
    taken = existing_issues()
    print(f"      {len(taken)} existing issues on remote")

    print("\n[4/4] Push drafts")
    search_root = ROOT / args.subset if args.subset else ROOT
    drafts = sorted(
        p for p in search_root.rglob("*.md")
        if p.name.lower() != "readme.md"
    )
    if not drafts:
        print("      (no drafts found)")
        return 0

    for path in drafts:
        try:
            draft = load_draft(path)
        except ValueError as exc:
            print(f"  !  skip ({exc})")
            continue
        push_draft(draft, taken, args.dry_run)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

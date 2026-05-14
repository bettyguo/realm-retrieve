---
title:    "chore(security): triage Dependabot's 85 flagged vulnerabilities (10 critical)"
labels:   ["dependencies", "help wanted"]
milestone: "v1.1"
state:    open
opened:   "2026-05-14"
---

## Symptom

Every push prints:

```
remote: GitHub found 85 vulnerabilities on bettyguo/realm-retrieve's default
remote: branch (10 critical, 18 high, 38 moderate, 19 low).
remote:      https://github.com/bettyguo/realm-retrieve/security/dependabot
```

The vast majority of these come from the **pinned versions** in
`requirements.txt` (a research repo's prerogative: frozen deps for
reproducibility). But 10 *critical* alerts deserve a look — at least to
classify them as "accepted risk because this is a research artifact" vs.
"genuinely should bump".

## Triage approach

For each critical/high alert:

1. Identify the affected dependency.
2. Determine if our code path actually triggers the vulnerable behaviour
   (most won't — we don't accept untrusted input).
3. If the answer is "no", dismiss with a clear "Risk accepted because: …"
   comment so future maintainers know we looked.
4. If the answer is "yes" or "uncertain", **bump the dep** (subject to
   reproducibility constraints — see CHANGELOG note).

## Constraints

- The paper's numbers were produced against the exact versions in
  `requirements.txt`. Bumping anything changes the surface area for the
  next reproducibility audit.
- We can be more aggressive in `pyproject.toml` ranges since those govern
  the *minimum*, not the *exact* version pip will install.

## Acceptance

- [ ] All 10 critical alerts have either a documented "risk accepted"
      dismissal or a fix PR.
- [ ] `requirements.txt` carries a comment block explaining the
      reproducibility-vs-freshness trade-off.
- [ ] A `SECURITY.md` section ("How we handle CVEs in pinned deps")
      points to this issue.

## Why help wanted

This is 80 % careful reading, 20 % code. Perfect contribution for someone
with security background who wants a low-stakes way to get involved.

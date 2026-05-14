# Issue tracker (offline index)

This file mirrors the canonical [GitHub issue
tracker](https://github.com/bettyguo/realm-retrieve/issues). It is
auto-derived from the markdown drafts under
[`.github/ISSUE_DRAFTS/`](.github/ISSUE_DRAFTS/) and lets you read the
project's roadmap without leaving the repo.

> **Push these to GitHub** via `bash scripts/seed_issues.sh` or `pwsh
> scripts/seed_issues.ps1` once you have `gh auth login` set up.

---

## Open

| #  | Title                                                                                                  | Labels                                | Milestone | Opened     |
|----|--------------------------------------------------------------------------------------------------------|---------------------------------------|-----------|------------|
| 6  | [chore(types): modernize type hints to PEP 604](.github/ISSUE_DRAFTS/open/06-modernize-type-hints.md) | `good first issue` `chore` `docs`     | v1.1      | 2026-05-13 |
| 7  | [test(ci): CPU smoke test from a wheel](.github/ISSUE_DRAFTS/open/07-cpu-smoke-test-pipeline.md)      | `good first issue` `testing` `ci`     | v1.1      | 2026-05-13 |
| 8  | [feat(retriever): `Retriever` Protocol](.github/ISSUE_DRAFTS/open/08-retriever-protocol.md)           | `enhancement` `help wanted` `design`  | v1.2      | 2026-05-13 |
| 9  | [research(rsus): multilingual entity entropy](.github/ISSUE_DRAFTS/open/09-multilingual-rsus.md)      | `research` `help wanted` `rsus`       | v2.0      | 2026-05-13 |
| 10 | [feat(demo): HuggingFace Space playground](.github/ISSUE_DRAFTS/open/10-huggingface-space.md)         | `enhancement` `help wanted` `demo`    | v2.0      | 2026-05-13 |

## Closed (shipped in v1.0)

| #  | Title                                                                                                       | Closed     | Closed by                                            |
|----|-------------------------------------------------------------------------------------------------------------|------------|------------------------------------------------------|
| 1  | [fix(evaluate): Hydra `config_path` mismatch](.github/ISSUE_DRAFTS/closed/01-hydra-config-path.md)         | 2026-04-22 | `configs/experiments/*` committed                     |
| 2  | [fix(policy): empty-episode crash in REINFORCETrainer](.github/ISSUE_DRAFTS/closed/02-train-policy-empty-episode.md) | 2026-04-28 | Guard in `models/policy.py`                          |
| 3  | [fix(imports): heavy submodule imports break CPU](.github/ISSUE_DRAFTS/closed/03-heavy-imports.md)         | 2026-05-05 | Lazy imports in `models/{reasoning_model,retriever,rsus}.py` |
| 4  | [feat(configs): ship `configs/experiments/*.yaml`](.github/ISSUE_DRAFTS/closed/04-missing-experiments-configs.md) | 2026-05-08 | Same release as #1                                   |
| 5  | [fix(rsus): validate α + β + γ ≈ 1](.github/ISSUE_DRAFTS/closed/05-rsus-weights-validation.md)            | 2026-05-12 | `ValueError` raised in `RSUSCalculator.__init__`     |

---

## Milestones

| Milestone | Due        | Theme                                          | Issues |
|-----------|------------|------------------------------------------------|--------|
| v1.0      | 2026-05-14 | SIGIR camera-ready release                     | #1–#5  |
| v1.1      | 2026-06-11 | Quality-of-life & onboarding polish            | #6, #7 |
| v1.2      | 2026-07-09 | Extensibility — pluggable retrievers          | #8     |
| v2.0      | 2026-09-03 | Multilingual + playground                      | #9, #10 |

## Looking for a way to contribute?

- ★ Start with anything tagged `good first issue` — these are scoped to a
  weekend or less and we'll mentor in review.
- 🧪 Try the [quickstart](README.md#-quickstart) and open an issue if
  anything is unclear.
- 🌍 If you work in zh / ja / es, comment on #9 — that's the one we most
  need outside help with.

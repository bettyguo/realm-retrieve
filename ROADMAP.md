# ReaLM-Retrieve Roadmap

A public roadmap for the next four months. Items below mirror the open issues
in [`.github/ISSUE_DRAFTS/`](.github/ISSUE_DRAFTS/) and live milestones on the
GitHub remote — they are the single source of truth and update automatically
as work lands.

> **Want to contribute?** Pick any `good first issue` from a milestone below,
> comment to claim it, and ship. Substantial items are tagged `help wanted` —
> open a Discussion before starting if you'd like to scope it together.

---

## v1.0 — *Camera-ready bugfix flight* · shipped 2026-05-14

Closed at the camera-ready cutoff. Items linked from
[`CHANGELOG.md`](CHANGELOG.md#100--2026-05-14):

| #   | Title                                                               | State  |
|----:|---------------------------------------------------------------------|--------|
| 1   | `fix(evaluate): Hydra config_path points to a non-existent directory` | closed |
| 2   | `fix(policy): REINFORCETrainer.train_step crashes on empty episodes` | closed |
| 3   | `fix(models): heavy backends must be lazy-imported`                  | closed |
| 4   | `feat(configs): ship the train_* Hydra configs`                      | closed |
| 5   | `fix(rsus): silently accept α + β + γ ≠ 1`                          | closed |

---

## v1.1 — *Polish & onboarding* · target 2026-06-11 (4 weeks)

Tighten the developer-experience loop after the burst of attention from
SIGIR. Everything here should be **doable in a weekend** by a new contributor.

| #   | Title                                                          | Labels                       | Effort  |
|----:|----------------------------------------------------------------|------------------------------|---------|
| 6   | PEP-604 type-hint migration                                    | `good first issue` `tech-debt` | ~2 h    |
| 7   | CPU import-smoke test in CI                                    | `good first issue` `ci`        | ~4 h    |

Stretch goals (no commitment, will move to v1.2 if we slip):

- Coverage badge on README once unit tests reach 80%.
- Pre-commit hook for `ruff format` (lock-step with the formatter version).

---

## v1.2 — *Plugin architecture & performance* · target 2026-07-09 (8 weeks)

Open the codebase to community extensions and chase the next round of
efficiency wins.

| #   | Title                                                          | Labels                  | Effort |
|----:|----------------------------------------------------------------|-------------------------|--------|
| 8   | `Retriever` Protocol + first-party BM25 / SPLADE adapters      | `help wanted`           | ~3 d   |

Stretch goals:

- Sub-quadratic step segmenter (`_token_to_char_position` is O(n²)).
- vLLM serving recipe (`make serve`).

---

## v2.0 — *Research extensions & live demo* · target 2026-09-03 (16 weeks)

The "this is more than a paper drop" milestone.

| #   | Title                                                          | Labels                  | Effort |
|----:|----------------------------------------------------------------|-------------------------|--------|
| 9   | Multilingual entity-entropy (zh / ja / es)                     | `research` `help wanted`| ~2 wk  |
| 10  | HuggingFace Space + interactive playground                     | `demo` `help wanted`    | ~1 wk  |

Stretch goals:

- Workshop-paper follow-up on the multilingual ablation.
- On-device variant (≤ 7B params end-to-end).

---

## Always-on backlog — `tech-debt`

Things we'll happily merge but won't block a release on. Each lives as a
separate issue:

- Strict `mypy` mode (currently lenient).
- Replace `defaultdict` imports we don't actually use.
- Move ASCII architecture diagram to a generated SVG so the figure stays
  in sync with the README.

---

## How dates are picked

We use a four-week cadence (one minor release per month) with a longer gap
into `v2.0` to give research items breathing room. **Dates slip** — that's
fine; the milestone burndown on GitHub is the canonical schedule.

## Questions?

Open a [Discussion](https://github.com/bettyguo/realm-retrieve/discussions) or
ping `@bettyguo` / `@hk950014` on an issue.

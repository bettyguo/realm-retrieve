# Contributing to ReaLM-Retrieve

Thanks for taking the time to contribute. ReaLM-Retrieve is an open
research project and we welcome bug reports, feature ideas, and pull
requests of all sizes.

This guide explains how to set up your environment, the conventions we
follow, and the steps to ship a change.

---

## 1. Quick setup

```bash
git clone https://github.com/bettyguo/realm-retrieve.git
cd realm-retrieve
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
make install-dev                                    # editable install + lint + pre-commit
make quickstart                                     # sanity check
```

`make install-dev` registers a [pre-commit](https://pre-commit.com/) hook
that runs `ruff`, `black`, and `isort` on every commit. CI will fail if
those tools find problems, so the hook saves you a round-trip.

---

## 2. Branch & PR flow

1. **Fork** the repo and create a topic branch off `main`:
   ```bash
   git checkout -b feat/short-description
   ```
2. **Write tests first** when the change is behavioural — `tests/`
   mirrors `src/realm_retrieve/`. Bug fixes should include a regression
   test that fails on `main`.
3. **Keep commits scoped.** Prefer many small commits to one mega-commit;
   we squash-merge so the PR title/body becomes the final commit message.
4. **Run `make ci` locally.** It runs lint + typecheck + tests in the
   same configuration CI uses.
5. **Open a PR** with the template — fill in the *Why*, the *What*, and
   the *Test plan*. Link related issues.
6. A maintainer will review within a few business days. Substantive
   review feedback is normal — please don't take it personally.

### Commit message style

Conventional Commits, lightly enforced:

```
feat(policy): add entropy bonus to REINFORCE loss
fix(rsus): handle zero-entity steps without ZeroDivision
docs(readme): clarify CPU-only quickstart
test(metrics): cover bootstrap CI edge cases
```

---

## 3. Coding conventions

- **Formatter:** `black` (line length 100). `make format` fixes everything.
- **Imports:** `isort` (`profile = black`), grouped stdlib / third-party / first-party.
- **Linter:** `ruff` with the rule set in `pyproject.toml`. Treat warnings as errors.
- **Types:** type-hint public functions and classes. We run `mypy` in
  *non-strict* mode for now — strict mode is a roadmap item.
- **Docstrings:** Google style. Public APIs need a one-line summary + an
  Args / Returns section.
- **Comments:** explain *why*, not *what*. The code already says what.
- **Logging:** use the `rich`-backed logger from `realm_retrieve.utils.logging`
  rather than `print`.

---

## 4. Tests

```bash
make test           # fast unit tests
make test-all       # everything, including slow + gpu-marked tests
pytest -k segment   # subset via pytest -k
```

Markers (`pyproject.toml`):

- `slow`        — > 10 s; skipped by default in CI fast lane.
- `gpu`         — requires CUDA; skipped on CPU runners.
- `integration` — touches the network or large checkpoints.

We aim for **≥ 80 % line coverage on `realm_retrieve/`**. New modules
should land with tests.

---

## 5. Reproducibility checklist

Research code is only useful if others can re-run it. When changing
anything that affects reported numbers, please:

- [ ] Seed every RNG you control (`torch`, `numpy`, `random`).
- [ ] Update / add a Hydra config under `configs/experiments/`.
- [ ] Re-run `make eval DATASET=musique` and paste the result table in
      the PR.
- [ ] If you changed the policy / segmenter checkpoints, regenerate them
      and bump the version in `CHANGELOG.md`.

---

## 6. Issue triage

We label issues as:

- `bug`            — something is wrong with the code or docs.
- `enhancement`    — a backwards-compatible improvement.
- `breaking`       — needs a major version bump.
- `good first issue` — scoped tightly enough for newcomers.
- `help wanted`    — we'd love a contribution here.

If you're not sure where to start, look for `good first issue`.

---

## 7. Code of Conduct

By participating you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md). Be kind, be patient.

---

Thanks again — every PR, issue, and ⭐ helps.

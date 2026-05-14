# Changelog

All notable changes to ReaLM-Retrieve will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- ​​​—

### Changed
- ​​​—

### Fixed
- ​​​—

---

## [1.0.0] — 2026-05-14

The SIGIR 2026 camera-ready release.

### Added
- **Reasoning step segmenter** — 3-layer transformer boundary classifier
  trained on 2,847 human-annotated DeepSeek-R1 traces (94.2 F1).
- **RSUS calculator** — three-signal step-level uncertainty score
  (verbalised, entity entropy, consistency).
- **Retrieval intervention policy** — REINFORCE-trained policy network
  with curriculum-learned cost coefficient (λ₁: 0.5 → 0.1).
- **ColBERTv2 + PLAID retriever wrapper** with optional WARP optimisations.
- **Unified LRM adapter** for DeepSeek-R1, OpenAI o1, and Qwen QwQ.
- **Evaluation suite** — EM / F1 / Sup-F1 / Evi-F1 plus IR metrics via
  `pytrec_eval` and paired-bootstrap significance testing.
- CPU-friendly **quickstart demo** (`examples/quickstart.py`).
- mkdocs documentation scaffold, Dockerfile, GitHub Actions CI.

### Reproduces
- MuSiQue: **EM 63.5 · F1 71.2 · 1.8 retrievals/q**
- HotpotQA: **F1 78.4 · 1.4 retrievals/q**
- 2WikiMHQA: **F1 74.9 · 1.6 retrievals/q**

[Unreleased]: https://github.com/bettyguo/realm-retrieve/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/bettyguo/realm-retrieve/releases/tag/v1.0.0

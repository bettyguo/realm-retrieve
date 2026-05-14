# ReaLM-Retrieve v1.0.0 — Camera-ready

The first stable release, accompanying the SIGIR '26 paper.

## Highlights

- **71.2 F1** on MuSiQue with only **1.8 retrieval calls per question** — +5.8 F1 over IRCoT, with 47 % fewer retrievals.
- Statistically significant at *p < 0.01* (paired bootstrap, 10K iter, Bonferroni-corrected) across **MuSiQue**, **HotpotQA**, and **2WikiMHQA**.
- **CPU-friendly quickstart** so newcomers can see the pipeline end-to-end in under 2 minutes with no GPU.
- **Lazy-import discipline** lets you `import realm_retrieve.models.*` without installing vLLM / ColBERT / spaCy.

## What's inside

- **ReasoningStepSegmenter** — 3-layer transformer boundary classifier (94.2 F1, 2,847 human-annotated traces).
- **RSUSCalculator** — three-signal step-level uncertainty score with validated `α + β + γ = 1` weight constraint.
- **RetrievalInterventionPolicy** — REINFORCE-trained policy network with curriculum-annealed cost coefficient `λ₁: 0.5 → 0.1`.
- **ColBERTRetriever** — ColBERTv2 + PLAID wrapper with WARP-style per-call caching.
- **Unified LRM adapter** — DeepSeek-R1, OpenAI o1, Qwen QwQ-32B all behind one interface.
- **Evaluation suite** — EM / F1 / Sup-F1 / Evi-F1 + `pytrec_eval` IR metrics + paired-bootstrap significance.

## Reproducibility

```bash
git clone https://github.com/bettyguo/realm-retrieve.git
cd realm-retrieve
pip install -e ".[all]"
python -m spacy download en_core_web_sm
make data        # ~68 GB, ~6 h
make train-all   # ~4 h seg + ~12.5 d policy on 8× A100
make eval        # ~2 h per benchmark
```

Expected numbers:

| Dataset    | F1 (Δ vs. IRCoT) | Retrievals/q (Δ vs. IRCoT) |
|------------|-----------------:|---------------------------:|
| MuSiQue    | **71.2 (+5.8)**  | **1.8 (−47 %)**            |
| HotpotQA   | **78.4 (+3.1)**  | **1.4 (−51 %)**            |
| 2WikiMHQA  | **74.9 (+4.7)**  | **1.6 (−43 %)**            |

If you can't reproduce within bootstrap-CI tolerance, [open a bug](https://github.com/bettyguo/realm-retrieve/issues/new/choose) — reproducibility regressions count as bugs.

## Bug fixes that landed in this release

| # | Title |
|---|---|
| [#26](https://github.com/bettyguo/realm-retrieve/issues/26) | `evaluate.py` Hydra `config_path` pointed to a non-existent directory |
| [#28](https://github.com/bettyguo/realm-retrieve/issues/28) | Heavy submodule imports broke CPU-only / API-only environments |
| [#29](https://github.com/bettyguo/realm-retrieve/issues/29) | `REINFORCETrainer.train_step` crashed on empty placeholder episodes |
| [#30](https://github.com/bettyguo/realm-retrieve/issues/30) | Missing `configs/experiments/{evaluate,train_segmentation,train_policy}.yaml` |
| [#31](https://github.com/bettyguo/realm-retrieve/issues/31) | RSUS silently accepted `(α, β, γ)` weights that didn't sum to 1 |

See [CHANGELOG.md § 1.0.0](https://github.com/bettyguo/realm-retrieve/blob/main/CHANGELOG.md#100--2026-05-14) for fix details.

## What's next

Five open issues across three milestones — see the [Roadmap](https://github.com/bettyguo/realm-retrieve/wiki/Roadmap).

- **v1.1** (due 2026-06-11): polish + onboarding. 2 `good first issue`s.
- **v1.2** (due 2026-07-09): plugin retriever protocol.
- **v2.0** (due 2026-09-03): multilingual support + HuggingFace Space demo.

If ReaLM-Retrieve helps your research, please cite the paper (BibTeX in [`CITATION.cff`](https://github.com/bettyguo/realm-retrieve/blob/main/CITATION.cff)) and ⭐ the repo — star count is our best signal for prioritising community work.

## Citation

```bibtex
@inproceedings{guo2026realmretrieve,
  title     = {When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models},
  author    = {Guo, Dongxin and Wu, Jikun and Yiu, Siu Ming},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {Melbourne, Australia},
  doi       = {10.1145/3805712.3809722}
}
```

## Acknowledgements

ReaLM-Retrieve builds on Stanford NLP's **ColBERTv2**, DeepSeek's **R1**, and UC Berkeley Sky's **vLLM**. Inspired by **IRCoT**, **FLARE**, **Self-RAG**, and **Search-R1**.

Made at HKU & Stellaris AI. Thanks to all the anonymous SIGIR reviewers whose feedback hardened the manuscript.

— Dongxin Guo, Jikun Wu, Siu Ming Yiu

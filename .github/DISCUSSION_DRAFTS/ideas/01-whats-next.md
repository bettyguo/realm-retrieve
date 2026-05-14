---
title:    "🧪 What should v1.2 / v2.0 prioritise? Drop your ideas here."
category: ideas
pin:      true
lock:     false
---

The [public roadmap](https://github.com/bettyguo/realm-retrieve/wiki/Roadmap)
has five issues across `v1.1`, `v1.2`, and `v2.0`. But what's there is
**what we think matters** — and we'd much rather build what *you* need.

## Already on the docket

`v1.1` (small + polish, due 2026-06-11):
- [#32 PEP-604 type hints](https://github.com/bettyguo/realm-retrieve/issues/32)
- [#33 CPU import smoke test in CI](https://github.com/bettyguo/realm-retrieve/issues/33)

`v1.2` (architecture, due 2026-07-09):
- [#34 `Retriever` Protocol + BM25 / SPLADE adapters](https://github.com/bettyguo/realm-retrieve/issues/34)

`v2.0` (research, due 2026-09-03):
- [#27 Multilingual entity-entropy (zh / ja / es)](https://github.com/bettyguo/realm-retrieve/issues/27)
- [#35 HuggingFace Space demo](https://github.com/bettyguo/realm-retrieve/issues/35)

## What we're curious about (low-confidence priors, please push back)

- **Streaming RSUS** — compute the verbalised confidence *during* generation
  rather than as a separate forward pass. Would it work?
- **Tool-calling integration** — should the retrieval-intervention policy
  also decide *what tool to call*, not just whether to retrieve?
- **Distillation** — can we compress segmenter + policy + LRM into a single
  ≤ 7B-parameter model that runs on a laptop?
- **Online learning** — update the policy from user feedback after deployment.
- **Cross-LRM transfer** — how much do RSUS weights need to be re-tuned per
  reasoning model?

## How to participate

- React 👍 on the bullets above if you'd find them useful.
- Reply with your own ideas — even half-baked ones; we like to brainstorm.
- If your idea is concrete enough to be a unit of work, **open an Issue**
  with the `enhancement` label. We'll fold it into a milestone.

We score-rank Ideas by 👍 count when promoting to a versioned milestone, so
your reaction is the loudest single signal we get.

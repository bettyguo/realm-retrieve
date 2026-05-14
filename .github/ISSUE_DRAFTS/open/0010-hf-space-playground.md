---
title:    "feat(demo): ship a HuggingFace Space + interactive playground"
labels:   ["enhancement", "demo", "help wanted"]
milestone: "v2.0"
state:    open
opened:   "2026-05-13"
---

## Goal

A public HuggingFace Space that lets visitors:

1. Type a multi-hop question.
2. Watch the reasoning chain stream in token-by-token.
3. See **per-step RSUS scores** rendered next to each step.
4. See the policy decision (`RETRIEVE` / `SKIP`) and the documents retrieved.
5. Compare against a vanilla **Single-RAG** baseline side-by-side.

The point is to turn the paper's central insight ("retrieve only when the
model is uncertain *mid-stream*") into something visitors can **feel** in 30
seconds.

## Constraints

- Must fit on a free-tier T4 — use `DeepSeek-R1-Distill-Qwen-1.5B` as the
  default LRM.
- Cold-start under 60 s.
- No private dataset dependencies — index is the public MuSiQue dev set.

## Proposed stack

- Gradio 4.x for UI (custom blocks for the stepwise visualization).
- `realm-retrieve` from PyPI once Issue #4's release pipeline is in place.
- Pre-built ColBERT index cached in the Space's `data/` dir.

## Acceptance

- [ ] Space at `https://huggingface.co/spaces/bettyguo/realm-retrieve` runs
      green on free tier.
- [ ] README's "Live demo" badge links to it.
- [ ] Recorded GIF of the demo replaces the placeholder hero banner.

## Hints

- See [`examples/quickstart.py`](../../../examples/quickstart.py) for the
  control flow you need to mirror.
- The toy module gives you a CPU fallback when the GPU queue is full — use it
  to keep the Space responsive on free tier.

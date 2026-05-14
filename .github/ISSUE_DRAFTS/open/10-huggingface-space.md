---
title:     "feat(demo): HuggingFace Space with an interactive playground"
labels:    ["enhancement", "help wanted", "demo"]
milestone: "v2.0 (2026-09-03)"
state:     open
opened:    "2026-05-13"
---

## Goal

A no-install [HuggingFace Space](https://huggingface.co/spaces) that lets a
visitor type a multi-hop question, watch the reasoning chain stream in, and
**see the policy fire** for steps with high RSUS — with the retrieved
passages rendered next to each step.

The point is reach: a working Space + a 30-second screen-recording embedded
in the README is the single highest-leverage thing for community signal
once the v1.0 paper drops.

## Sketch

- Gradio frontend (Blocks API, streaming) — left pane shows the reasoning
  chain coloured by RSUS, right pane shows retrieved passages.
- Backend uses the `[api]` extra (OpenAI / Anthropic) so the Space can run
  on a CPU instance — no GPU bills.
- ColBERT-replacement: tiny BM25 index over a 50-document Wikipedia subset
  (built at container start, free).
- All three uncertainty signals exposed as collapsible "Why did it
  retrieve?" panels.

## Sub-tasks

- [ ] `spaces/playground/` directory with `app.py`, `requirements.txt`,
      `Dockerfile`, `README.md` Space metadata.
- [ ] Tiny BM25 over 50 Wikipedia passages (cached to disk).
- [ ] Switchable model dropdown — OpenAI o4-mini, Claude Haiku 4.5,
      DeepSeek-R1 via Together AI.
- [ ] Embedded link + screen recording back in the main README hero.
- [ ] Cost / rate-limit guardrails (max tokens, max retrievals/q).

## Acceptance

- [ ] Public URL listed in the README badges section.
- [ ] End-to-end response < 6 s for a typical 2-hop question.
- [ ] The Space goes through the same pre-commit hooks as the main repo.

If you've shipped Gradio Blocks before — this would be a great PR and we'll
credit you on the landing page.

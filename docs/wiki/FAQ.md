# FAQ

Recurring questions from the community. If yours isn't here, ask on
[Discussions](https://github.com/bettyguo/realm-retrieve/discussions) and
we'll fold the good ones back into this page.

---

## I don't have 8× A100s. Can I still use this?

**Yes.** The two trained checkpoints — segmenter (~6 MB) and policy
(~40 MB) — are *tiny*. The reasoning model is the only GPU-hungry piece.

Three sizing options for the LRM:

| Setup                              | What you need              | Reproduces paper? |
|------------------------------------|----------------------------|-------------------|
| `DeepSeek-R1-Distill-Qwen-1.5B`    | 1 × 12 GB GPU              | No — smaller model, expect ~10 F1 below.  |
| `DeepSeek-R1-Distill-Qwen-32B`     | 1 × 80 GB or 2 × 48 GB     | **Yes** — paper's main config.            |
| OpenAI `o1` / `o1-preview`         | API key, no GPU            | Different model family; expect ±2 F1.     |

The CPU [quickstart](https://github.com/bettyguo/realm-retrieve/blob/main/examples/quickstart.py)
uses a toy LRM and BM25 retrieval — no GPU needed, just for understanding
the pipeline shape.

---

## How does RSUS differ from FLARE's token-probability trigger?

FLARE triggers on single low-probability tokens. That fires constantly during
exploratory reasoning ("hmm", "wait", "let me think") which is *not* the same
as factual uncertainty.

RSUS aggregates **three orthogonal signals** at the **step** level
(~127 tokens), so it only fires when the model is collectively uncertain
about a *claim*, not just stylistic word choice. Net effect: fewer false
retrieval triggers, higher F1 per call.

---

## Can I replace ColBERTv2 with my own retriever?

In v1.0 it requires a small subclass. In v1.2 we'll ship a `typing.Protocol`
that lets you inject any callable with the signature
`retrieve(query: str, k: int) -> list[dict]` — tracked by
[#34](https://github.com/bettyguo/realm-retrieve/issues/34). Comment on the
issue if your timeline can't wait for v1.2 and we'll backport.

---

## Does it work for code, math, or non-English questions?

- **Code / math**: yes with `β=0` (skip the entity-entropy signal — there
  are no named entities to count) and falling back on `U_verb` + `U_cons`.
  We have unpublished ablations on HumanEval+/MATH that look promising;
  patches welcome.
- **Non-English**: needs a per-language NER pipeline. Active research item
  tracked by [#27](https://github.com/bettyguo/realm-retrieve/issues/27).
  We've seeded the zh/ja/es spaCy pipelines but the validation pass is
  open work.

---

## Why REINFORCE and not PPO / GRPO?

The action space is binary (retrieve / continue), the episodes are short
(< 10 steps typically), and the reward is dense (F1 is per-question, latency
is per-call). REINFORCE with a moving baseline converges in 50K steps and
gives stable behaviour. PPO's clipping is overkill at this action-space size;
GRPO's group structure assumes parallel trajectories we don't have. Happy to
revisit — open a discussion thread if you're seeing PPO win on a derivative
benchmark.

---

## Can I cite the camera-ready before SIGIR '26 happens?

Yes — the DOI `10.1145/3805712.3809722` is live as soon as ACM accepts the
camera-ready, which they have. BibTeX is in
[`CITATION.cff`](https://github.com/bettyguo/realm-retrieve/blob/main/CITATION.cff)
and on the [Citation](Citation) wiki page.

---

## My CI fails on import — what gives?

Almost certainly Issue [#3](https://github.com/bettyguo/realm-retrieve/issues/28)
rearing its head again on a new module. We fixed the three known cases
(`vllm`, `colbert`, `spacy`) by lazy-importing inside the constructors. If
you're hitting a *new* one, please open a bug and tag it `imports` — we'll
backport the lazy-import fix.

(There's a CI smoke test tracked by
[#33](https://github.com/bettyguo/realm-retrieve/issues/33) that's meant to
catch this category automatically — `good first issue` if you want to
land it.)

---

## How do I get involved?

- Browse [`good first issue`](https://github.com/bettyguo/realm-retrieve/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
- Read [CONTRIBUTING.md](https://github.com/bettyguo/realm-retrieve/blob/main/CONTRIBUTING.md).
- For larger research extensions, open a [Discussion](https://github.com/bettyguo/realm-retrieve/discussions/categories/ideas)
  first so we can scope.

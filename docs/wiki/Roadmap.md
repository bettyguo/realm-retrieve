# Roadmap

Public timeline through SIGIR '26 (July 20-24, Melbourne). All milestones map
to real GitHub milestones with realistic due dates.

---

## v1.0 — Camera-ready  (shipped 2026-05-14) ✅

The SIGIR '26 release. 5 closed bug fixes, baseline 71.2 F1 on MuSiQue, full
ColBERT+PLAID stack, REINFORCE policy checkpoints. See
[CHANGELOG.md § 1.0.0](https://github.com/bettyguo/realm-retrieve/blob/main/CHANGELOG.md).

---

## v1.1 — Polish & onboarding  (due 2026-06-11)

Two open `good first issue`s. Goal: lower the bar to first PR.

- **[#32](https://github.com/bettyguo/realm-retrieve/issues/32)** — migrate `Optional[X]` / `List[X]` to PEP-604 syntax.
- **[#33](https://github.com/bettyguo/realm-retrieve/issues/33)** — CI smoke test that imports every production module from a wheel (regression coverage for [#3](https://github.com/bettyguo/realm-retrieve/issues/28)).

If you're new to the codebase, **start here**.

---

## v1.2 — Plugin architecture  (due 2026-07-09)

One issue, hefty. Lets users swap the retrieval backend without subclassing.

- **[#34](https://github.com/bettyguo/realm-retrieve/issues/34)** — extract a `Retriever` Protocol; ship a `BM25Retriever` reference impl alongside the existing ColBERT path.

---

## v2.0 — Research extensions  (due 2026-09-03)

Bigger, more research-flavoured. Co-authorship on follow-up workshop papers
on offer for substantive contributions.

- **[#27](https://github.com/bettyguo/realm-retrieve/issues/27)** — multilingual entity-entropy support (zh / ja / es).
- **[#35](https://github.com/bettyguo/realm-retrieve/issues/35)** — HuggingFace Space + interactive playground.

---

## Always-on: `tech-debt` milestone

Items that don't block any specific release. Anyone is welcome to pick them
up. Browse via the [milestone view](https://github.com/bettyguo/realm-retrieve/milestone/5)
or the [`tech-debt`](https://github.com/bettyguo/realm-retrieve/labels/tech-debt)
label.

---

## Stretch items — not yet tracked

These are on the team's mind but don't have a milestone or issue yet. Open a
[Discussion](https://github.com/bettyguo/realm-retrieve/discussions/categories/ideas)
to bring one forward.

- **Streaming RSUS**: compute `U_verb` / `U_ent` *during* generation rather than
  post-hoc, to avoid the second forward pass.
- **vLLM-native deployment recipe**: single-command `realm-retrieve serve`
  that spins up the full pipeline behind an OpenAI-compatible API.
- **On-device variant**: distill the policy + segmenter into a single ≤ 7 B
  model. Realistic candidate for laptop deployment.
- **Hard-negative mining for `U_ent`**: improve entropy calibration on
  long-tail entities by mining contrastive negatives during retrieval-index
  build.

---

## How to influence the roadmap

1. **Vote with reactions.** 👍 on issues matters — we sort by reaction count
   when promoting from `tech-debt` to a versioned milestone.
2. **Open a Discussion** under *Ideas*. Things that get traction get scoped
   into an issue and a milestone.
3. **Submit a draft PR.** Even half-finished, it forces a real
   conversation about feasibility and design.

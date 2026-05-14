# Welcome to the ReaLM-Retrieve Wiki

This wiki is the **long-form knowledge base** for ReaLM-Retrieve — the
counterpart to the README (which is the elevator pitch) and the SIGIR '26
paper (which is the formal contribution). Use it for:

- **Design rationale** that didn't fit in the paper.
- **Operational recipes** (training tips, GPU footprint, retrieval index sizing).
- **Glossary** of paper-specific terminology.
- **FAQ** of community questions that recur.

> **Source of truth:** every wiki page mirrors a file in
> [`docs/wiki/`](https://github.com/bettyguo/realm-retrieve/tree/main/docs/wiki).
> Pull requests welcome.  The two stay in sync via
> [`scripts/sync_wiki.sh`](https://github.com/bettyguo/realm-retrieve/blob/main/scripts/sync_wiki.sh).

---

## Navigate

| Page                                      | What's in it                                                               |
|-------------------------------------------|----------------------------------------------------------------------------|
| [Architecture](Architecture)              | Step-by-step walkthrough of the four-stage pipeline.                       |
| [Reproducibility](Reproducibility)        | Datasets, training recipes, evaluation harness, paired-bootstrap details.  |
| [FAQ](FAQ)                                | Recurring community questions with terse answers.                          |
| [Glossary](Glossary)                      | Paper-specific terms (RSUS, U_verb, U_ent, U_cons, …) defined in one place.|
| [Roadmap](Roadmap)                        | Public milestones with realistic timelines and `good first issue`s.        |
| [Citation](Citation)                      | Plain BibTeX + `CITATION.cff` for your reference manager.                  |

---

## 30-second pitch

Large reasoning models (DeepSeek-R1, o1, QwQ) think for **thousands of tokens**
before answering. Classic RAG retrieves **once**, up front. **ReaLM-Retrieve**
learns where inside the chain of thought retrieval actually helps, and skips
the rest:

> **+5.8 F1** on MuSiQue, **47 % fewer retrieval calls** than IRCoT,
> **2.1× better accuracy-per-call ratio**. Significant at *p < 0.01*.

The full pipeline:

```
question
   │
   ▼
LRM → reasoning chain (~10K tokens)
   │
   ▼ ① ReasoningStepSegmenter
steps_i  (avg 127 tokens)
   │
   ▼ ② RSUS = α·U_verb + β·U_ent + γ·U_cons
score_i ∈ [0, 1]
   │
   ▼ ③ π(retrieve | state)   ← REINFORCE-trained policy
should_retrieve_i ∈ {0, 1}
   │
   ▼ ④ ColBERTv2 + PLAID
top-k passages
   │
   ▼ context fusion → final answer
```

---

## Where to ask things

- **Bug?** Open an [issue](https://github.com/bettyguo/realm-retrieve/issues/new/choose).
- **Open-ended question or idea?** [Discussions](https://github.com/bettyguo/realm-retrieve/discussions).
- **Security report?** [SECURITY.md](https://github.com/bettyguo/realm-retrieve/blob/main/SECURITY.md) (do **not** open a public issue).
- **Citing the work?** [Citation](Citation) page.

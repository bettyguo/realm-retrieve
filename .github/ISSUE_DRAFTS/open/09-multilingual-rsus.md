---
title:     "research(rsus): multilingual entity-entropy support (zh / ja / es)"
labels:    ["research", "help wanted", "rsus"]
milestone: "v2.0 (2026-09-03)"
state:     open
opened:    "2026-05-13"
---

## Why

`U_ent` (entity-coverage entropy) currently uses spaCy's `en_core_web_sm`
NER, which silently returns an empty span set for non-English reasoning
traces. That collapses RSUS to `α·U_verb + γ·U_cons`, and on our internal
zh / ja / es benchmarks the policy under-retrieves by ~30 %.

We hint at this in the [FAQ](../../../README.md#-faq) but never properly
shipped multilingual support.

## What needs to happen

1. Add a thin `EntityExtractor` abstraction so we can swap implementations
   per language:
   ```python
   class EntityExtractor(Protocol):
       def entities(self, text: str) -> list[str]: ...
   ```
2. Ship three concrete implementations:
   - `SpacyExtractor("en_core_web_sm")` — current behaviour.
   - `SpacyExtractor("xx_ent_wiki_sm")` — multilingual fallback.
   - `StanzaExtractor(lang="zh")` — for Chinese where spaCy's coverage is
     weak (we tested in §6.3 of the paper).
3. Auto-detect the language with `langid` (already pulled by `datasets`) and
   pick the right extractor.
4. Re-train weights `(α, β, γ)` on a Chinese subset of MuSiQue translated by
   GPT-4o (kept under `data/i18n/`) and report results in a follow-up paper.

## Acceptance

- [ ] `EntityExtractor` lives next to `Retriever` (see #8) and is documented.
- [ ] All three implementations land with tests.
- [ ] Eval report on the zh subset shows F1 within 2 points of English (or
      the gap is honestly reported in a results table).

## Hard problems we want input on

- **Quote / proper-noun tokenisation** in CJK. spaCy's tokeniser is OK; Stanza's
  is better but heavy.
- **Sentence boundary detection** in mid-reasoning steps that mix languages
  (code switches in technical reasoning are common).

If you've worked on multilingual NER or want to use this as a research
extension, **please leave a comment** — happy to scope and co-author.

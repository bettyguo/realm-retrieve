---
title:    "research(rsus): multilingual entity-entropy support (zh / ja / es)"
labels:   ["research", "help wanted", "rsus"]
milestone: "v2.0"
state:    open
opened:   "2026-05-13"
---

## Background

The RSUS entity-entropy signal (`U_ent`) currently uses spaCy's
`en_core_web_sm` NER model. On Chinese / Japanese / Spanish multi-hop QA
benchmarks (`Mr.TyDi-zh`, `JEMHopQA`, `XOR-AttriQA-es`) we observe entity
recall drop by 20-40 pp because either:

1. The English NER doesn't tag CJK proper nouns at all, or
2. Spanish entities are tagged but with the wrong type (`ORG` instead of
   `PERSON`), so the `U_ent` summation skips them.

This caps RSUS quality in non-English settings.

## Hypothesis

A simple per-language NER stack — `zh_core_web_sm` / `ja_core_news_sm` /
`es_core_news_sm` — plus language detection at the step level should recover
most of the gap with no architectural changes.

## What we'd like

1. **Language detection** at step granularity (e.g. `lingua-py`, falling back
   to `langdetect`).
2. A `MultilingualRSUSCalculator` that lazy-loads the appropriate spaCy
   pipeline on first use.
3. Reproduce numbers on at least one non-English benchmark and PR a small
   `results-multilingual.md`.
4. RSUS weights `(α, β, γ)` may need re-tuning per language — please report
   what you find.

## Acceptance

- [ ] At least one non-English benchmark beats the English-NER fallback by
      `>= 3 F1`.
- [ ] No regression on MuSiQue / HotpotQA / 2WikiMHQA (within bootstrap CI).
- [ ] New tests in `tests/test_multilingual_rsus.py` mock the spaCy pipeline
      so CI stays fast.

## Notes

This is a real research contribution — happy to add you as a co-author on a
follow-up workshop paper.

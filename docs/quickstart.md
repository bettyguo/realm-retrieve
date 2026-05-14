# Quickstart

This page walks through `examples/quickstart.py` — the same pipeline as the
production system, but with toy stand-ins for the heavy components so it runs
on CPU in under 2 seconds.

```bash
make quickstart
```

## What's happening, in 5 boxes

1. **Toy corpus.** Twelve hand-written passages cover the five demo questions
   plus a few distractors (see `realm_retrieve.toy.demo_corpus`).

2. **Toy LRM.** `ToyReasoningModel.generate_reasoning(...)` deterministically
   returns a two-step chain — one hedging step, one verification step.

3. **RSUS.** `ToyPipeline.rsus(step)` combines a verbalised-uncertainty
   proxy (hedging words), an entity-entropy proxy (capitalised-token density),
   and a length-based consistency proxy.

4. **Policy.** A threshold on the RSUS score — `should_retrieve = rsus ≥ θ`.
   In the production system this threshold is replaced by a REINFORCE-trained
   neural policy.

5. **Retrieval & answer.** When the policy fires, the BM25-based
   `ToyRetriever` returns top-3 passages and the toy LRM extracts an answer
   from the highest-overlap passage.

## Output (truncated)

```
[1/5] OK   rsus=['0.18']             retr=0  → '1989'        (gold '1989')
[2/5] OK   rsus=['0.72', '0.14']     retr=1  → 'Beijing'     (gold 'Beijing')
[3/5] OK   rsus=['0.72', '0.14']     retr=1  → 'Stockholm'   (gold 'Stockholm')
…
EM 80.0  |  F1 86.7  |  retrievals/q 1.60
```

## Going from toy → real

A single import swap is all it takes:

```diff
- from realm_retrieve.toy import ToyRetriever, ToyReasoningModel, ToyPipeline
+ from realm_retrieve import ColBERTRetriever, create_reasoning_model
```

…then point `ColBERTRetriever` at your PLAID index and `create_reasoning_model`
at DeepSeek-R1 (or any other LRM). See [Inference](inference.md) for the
full recipe.

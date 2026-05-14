---
title:    "🛟 Troubleshooting starter — common errors and how to file a good Q"
category: q-and-a
pin:      true
lock:     false
---

Before opening a new Q&A thread, please skim this checklist — most
questions resolve in under 30 seconds with the right pointer.

## 1. Did you check the FAQ?

The [FAQ](https://github.com/bettyguo/realm-retrieve/wiki/FAQ) covers the
single most common questions:

- "I don't have 8× A100s. Can I still use this?"
- "How is RSUS different from FLARE?"
- "Can I swap ColBERT for my own retriever?"
- "Does it work for code / math / non-English?"
- "Why REINFORCE and not PPO / GRPO?"

If yours isn't there, post it here and we'll fold the answer back into the
wiki for the next person.

## 2. Did the error give you a hint?

| Error                                                   | Most likely fix                                                  |
|---------------------------------------------------------|------------------------------------------------------------------|
| `MissingConfigException: Cannot find primary config 'evaluate'` | You're on a stale commit — update past [#26](https://github.com/bettyguo/realm-retrieve/issues/26). |
| `OSError [E050] Can't find model 'en_core_web_sm'`     | `python -m spacy download en_core_web_sm` (see [#31](https://github.com/bettyguo/realm-retrieve/issues/31)). |
| `RuntimeError: stack expects a non-empty TensorList`   | Update past [#29](https://github.com/bettyguo/realm-retrieve/issues/29); the empty-rollout fix landed in v1.0. |
| `ImportError: cannot import name 'LLM' from 'vllm'`    | Update past [#28](https://github.com/bettyguo/realm-retrieve/issues/28); heavy imports are now lazy. |
| `ValueError: RSUS weights must sum to 1`               | Working as intended — check `(α, β, γ)` in your Hydra config.    |
| `CUDA out of memory` during retrieval                  | Lower `tensor_parallel_size` or use `DeepSeek-R1-Distill-Qwen-1.5B`. |

## 3. How to file a good question

Copy-paste this template:

```
**What I'm trying to do**
<one sentence>

**What I ran**
```bash
# exact command(s)
```

**What I expected**
<one sentence>

**What I got**
```
# full traceback or unexpected output
```

**Environment**
- OS:
- Python:
- PyTorch / CUDA / GPU:
- ReaLM-Retrieve commit/version: (e.g. v1.0.0 or commit 1c5b8d2)
- Reasoning model:
```

Questions in this format get answered ~5× faster than open-ended ones.

## 4. When to file an Issue instead

- If you've identified an actual **bug** in our code → [Issue](https://github.com/bettyguo/realm-retrieve/issues/new?template=bug_report.yml).
- If you have a **specific feature request** with a concrete API in mind → [Issue](https://github.com/bettyguo/realm-retrieve/issues/new?template=feature_request.yml).
- If you're not sure → ask here first, we'll redirect.

## Maintainer turnaround

We typically respond within **3 business days**. If we've left you hanging
longer, feel free to bump with a polite ping — we'd rather get to you late
than not at all.

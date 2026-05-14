# ReaLM-Retrieve

**Adaptive retrieval for Large Reasoning Models** — SIGIR 2026.

This site is the reference manual for the [ReaLM-Retrieve](https://github.com/bettyguo/realm-retrieve)
library. For the elevator pitch and a 2-minute demo, see the
[project README](https://github.com/bettyguo/realm-retrieve#readme).

## Why it exists

Large reasoning models (DeepSeek-R1, OpenAI o1, Qwen QwQ) generate
**multi-thousand-token chains of thought** before answering. Plugging classic
RAG into that pipeline turns out to be much harder than it looks:

- Retrieving **once** before reasoning hands the model stale or irrelevant
  evidence — the actual knowledge gap appears mid-stream.
- Retrieving **every sentence** (IRCoT, FLARE) burns latency and pollutes the
  context with noise.

ReaLM-Retrieve learns *where* in the reasoning chain external evidence will
actually help, and skips the rest.

## What's in this manual

| Section | What you'll find |
|---------|------------------|
| [Installation](install.md) | CPU / GPU / Docker install paths |
| [Quickstart](quickstart.md) | The 2-minute CPU demo |
| [Inference](inference.md) | Plug your own LRM and retriever |
| [Concepts → RSUS](concepts/rsus.md) | The 3-signal uncertainty score |
| [Concepts → Policy](concepts/policy.md) | The REINFORCE retrieval policy |
| [API reference](reference/api.md) | Auto-generated docstrings |
| [Contributing](contributing.md) | Dev loop and PR etiquette |

## Citation

```bibtex
@inproceedings{guo2026realmretrieve,
  title     = {When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models},
  author    = {Guo, Dongxin and Wu, Jikun and Yiu, Siu Ming},
  booktitle = {SIGIR '26},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3805712.3809722}
}
```

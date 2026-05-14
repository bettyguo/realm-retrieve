# Citation

If ReaLM-Retrieve helps your research, please cite the SIGIR '26 paper. A
plain ⭐ on the repo also helps a lot — it's our best signal for prioritising
community work.

---

## BibTeX

```bibtex
@inproceedings{guo2026realmretrieve,
  title        = {When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models},
  author       = {Guo, Dongxin and Wu, Jikun and Yiu, Siu Ming},
  booktitle    = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '26)},
  year         = {2026},
  publisher    = {ACM},
  address      = {Melbourne, Australia},
  doi          = {10.1145/3805712.3809722},
  url          = {https://doi.org/10.1145/3805712.3809722}
}
```

---

## Plain text

> Dongxin Guo, Jikun Wu, and Siu Ming Yiu. 2026. When to Retrieve During
> Reasoning: Adaptive Retrieval for Large Reasoning Models. In *Proceedings
> of the 49th International ACM SIGIR Conference on Research and Development
> in Information Retrieval (SIGIR '26)*, July 20–24, 2026, Melbourne,
> Australia. ACM. https://doi.org/10.1145/3805712.3809722

---

## CITATION.cff

For reference managers that respect the
[Citation File Format](https://citation-file-format.github.io/), the
canonical file is at
[`/CITATION.cff`](https://github.com/bettyguo/realm-retrieve/blob/main/CITATION.cff).
GitHub renders a **Cite this repository** button in the sidebar automatically.

---

## Related work to cite alongside

The closest baselines from §2 of the paper:

```bibtex
@inproceedings{trivedi-etal-2023-interleaving,    # IRCoT
  author    = {Trivedi, Harsh and others},
  title     = {Interleaving Retrieval with Chain-of-Thought Reasoning ...},
  booktitle = {ACL},
  year      = {2023},
}

@inproceedings{jiang-etal-2023-active,            # FLARE
  author    = {Jiang, Zhengbao and others},
  title     = {Active Retrieval Augmented Generation},
  booktitle = {EMNLP},
  year      = {2023},
}

@inproceedings{asai-etal-2024-selfrag,            # Self-RAG
  author    = {Asai, Akari and others},
  title     = {Self-RAG: Learning to Retrieve, Generate and Critique through Self-Reflection},
  booktitle = {ICLR},
  year      = {2024},
}

@misc{jin-etal-2025-searchr1,                     # Search-R1
  author    = {Jin, Bowen and others},
  title     = {Search-R1: Training LLMs to Reason and Leverage Search Engines with RL},
  year      = {2025},
  eprint    = {2503.09516},
  archivePrefix = {arXiv},
}
```

See the [paper's `references.bib`](https://github.com/bettyguo/realm-retrieve/blob/main/paper/references.bib)
for the full bibliography.

---

## Acknowledgements (for derivative work)

If you build on ReaLM-Retrieve, please also acknowledge the upstream
retrieval and reasoning-model stacks:

- **ColBERTv2 / PLAID** — Stanford NLP, MIT-licensed.
- **DeepSeek-R1** — DeepSeek AI, MIT-licensed.
- **vLLM** — UC Berkeley Sky Computing Lab, Apache-2.0.

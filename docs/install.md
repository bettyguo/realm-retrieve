# Installation

ReaLM-Retrieve targets **Python 3.10 or 3.11** on Linux. macOS and Windows
work for the CPU quickstart; full training currently requires Linux + CUDA.

## CPU / quickstart

```bash
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
realm-quickstart                 # or:  make quickstart
```

## Full GPU stack (training + serving)

```bash
pip install -e ".[all]"
python -m spacy download en_core_web_sm
```

The `[all]` extra is the union of:

| Extra | Bundles                                                             |
|-------|----------------------------------------------------------------------|
| `serve` | vLLM, ColBERTv2, faiss-gpu, Ray                                   |
| `api`   | OpenAI, Anthropic                                                 |
| `train` | stable-baselines3, gymnasium, W&B, TensorBoard, Accelerate, PEFT  |
| `dev`   | pytest, ruff, black, isort, mypy, pre-commit                      |
| `docs`  | mkdocs, mkdocs-material, mkdocstrings                             |

## Docker

```bash
docker build -t realm-retrieve .
docker run --rm -it realm-retrieve            # runs the CPU quickstart
docker run --gpus all --rm -it realm-retrieve  bash
```

## Verifying the install

```bash
realm-retrieve version
realm-retrieve quickstart
make test
```

# =============================================================================
# ReaLM-Retrieve — developer Makefile
# =============================================================================
# Targets are grouped by lifecycle stage. Run `make help` for an overview.
# =============================================================================

SHELL          := /bin/bash
PYTHON         ?= python
PIP            ?= $(PYTHON) -m pip
PKG            := realm_retrieve
SRC            := src/$(PKG)
TESTS          := tests
DATA_DIR       ?= data
INDEX_DIR      ?= $(DATA_DIR)/indices/colbert
CKPT_DIR       ?= checkpoints
DATASET        ?= musique

# -----------------------------------------------------------------------------
# Meta
# -----------------------------------------------------------------------------
.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_.-]+:.*?## / { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
.PHONY: install
install:  ## Install runtime deps only.
	$(PIP) install -e .

.PHONY: install-dev
install-dev:  ## Install dev tooling (pytest, ruff, black, mypy, pre-commit, ...).
	$(PIP) install -e ".[dev]"
	$(PYTHON) -m pre_commit install || true

.PHONY: install-all
install-all:  ## Install everything (serve + api + train + dev + docs).
	$(PIP) install -e ".[all]"
	$(PYTHON) -m spacy download en_core_web_sm

# -----------------------------------------------------------------------------
# Lint / format / type-check
# -----------------------------------------------------------------------------
.PHONY: fmt format
fmt: format  ## Alias for format.
format:  ## Auto-format with black + isort + ruff.
	$(PYTHON) -m black $(SRC) $(TESTS) evaluate.py train_policy.py train_segmentation.py
	$(PYTHON) -m isort $(SRC) $(TESTS) evaluate.py train_policy.py train_segmentation.py
	$(PYTHON) -m ruff check --fix $(SRC) $(TESTS)

.PHONY: lint
lint:  ## Run linters (ruff + black --check + isort --check).
	$(PYTHON) -m ruff check $(SRC) $(TESTS)
	$(PYTHON) -m black --check $(SRC) $(TESTS)
	$(PYTHON) -m isort --check-only $(SRC) $(TESTS)

.PHONY: typecheck
typecheck:  ## Static type-check with mypy.
	$(PYTHON) -m mypy $(SRC)

# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
.PHONY: test
test:  ## Run unit tests with coverage.
	$(PYTHON) -m pytest $(TESTS) -m "not slow and not gpu"

.PHONY: test-all
test-all:  ## Run every test (slow + gpu included).
	$(PYTHON) -m pytest $(TESTS)

.PHONY: ci
ci: lint typecheck test  ## Full CI pipeline locally.

# -----------------------------------------------------------------------------
# Demos
# -----------------------------------------------------------------------------
.PHONY: quickstart
quickstart:  ## Run a CPU-friendly toy demo of the full pipeline.
	$(PYTHON) examples/quickstart.py

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
.PHONY: data
data:  ## Download MuSiQue + HotpotQA + 2WikiMHQA and build ColBERT indices.
	$(PYTHON) -m realm_retrieve.scripts.download_data --datasets musique hotpotqa 2wikimhqa --out $(DATA_DIR)
	$(PYTHON) -m realm_retrieve.scripts.build_index    --dataset $(DATASET) --out $(INDEX_DIR)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
.PHONY: train-segmentation
train-segmentation:  ## Train the step-boundary classifier.
	$(PYTHON) train_segmentation.py

.PHONY: train-policy
train-policy:  ## Train the REINFORCE retrieval policy.
	$(PYTHON) train_policy.py

.PHONY: train-all
train-all: train-segmentation train-policy  ## Train both modules end-to-end.

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
.PHONY: eval
eval:  ## Evaluate on $(DATASET) (override with `make eval DATASET=hotpotqa`).
	$(PYTHON) evaluate.py dataset=$(DATASET)

# -----------------------------------------------------------------------------
# Docs
# -----------------------------------------------------------------------------
.PHONY: docs
docs:  ## Build documentation site with mkdocs.
	$(PYTHON) -m mkdocs build

.PHONY: docs-serve
docs-serve:  ## Serve docs locally with hot-reload.
	$(PYTHON) -m mkdocs serve

# -----------------------------------------------------------------------------
# Housekeeping
# -----------------------------------------------------------------------------
.PHONY: clean
clean:  ## Remove build artifacts and caches.
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

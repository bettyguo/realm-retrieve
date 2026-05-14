"""Verify that the lazy package import works in a CPU-only environment.

These tests are critical — without lazy loading, ``import realm_retrieve`` on
a CI runner that lacks vLLM / ColBERT / spaCy would explode at import time.
"""

from __future__ import annotations

import importlib


def test_top_level_import_does_not_pull_heavy_deps():
    """Importing the package must not transitively import vllm or colbert."""
    import sys

    # Forget anything already loaded to make the assertion meaningful.
    for mod in list(sys.modules):
        if mod.startswith("realm_retrieve"):
            del sys.modules[mod]

    importlib.import_module("realm_retrieve")

    # Heavy frameworks must not have been loaded as a side-effect.
    assert "vllm" not in sys.modules
    assert "colbert" not in sys.modules
    assert "openai" not in sys.modules


def test_toy_module_is_self_contained():
    """``realm_retrieve.toy`` must import cleanly on CPU with no heavy deps."""
    mod = importlib.import_module("realm_retrieve.toy")
    assert hasattr(mod, "ToyPipeline")
    assert hasattr(mod, "ToyRetriever")
    assert hasattr(mod, "demo_corpus")


def test_evaluation_metrics_importable():
    mod = importlib.import_module("realm_retrieve.evaluation.metrics")
    assert hasattr(mod, "compute_f1")


def test_version_attribute_present():
    import realm_retrieve

    assert isinstance(realm_retrieve.__version__, str)
    assert realm_retrieve.__version__.count(".") >= 1

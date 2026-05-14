"""Tests for ``REINFORCETrainer`` (Issue #2).

The empty-episode short-circuit is the regression we care about; the rest is
construction-side sanity.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from realm_retrieve.models.policy import REINFORCETrainer, RetrievalInterventionPolicy


@pytest.fixture
def trainer() -> REINFORCETrainer:
    policy = RetrievalInterventionPolicy(
        embedding_dim=8,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        retrieval_threshold=0.5,
    )
    return REINFORCETrainer(policy=policy, device="cpu", learning_rate=1e-3)


def test_empty_episode_does_not_crash(trainer: REINFORCETrainer) -> None:
    """Regression for Issue #2."""
    metrics = trainer.train_step(
        states=[],
        actions=[],
        f1_score=0.7,
        num_retrievals=2,
        total_latency=0.5,
    )

    assert metrics["skipped_empty_episode"] is True
    assert metrics["loss"] == 0.0
    assert metrics["reward"] == pytest.approx(0.7 - 0.5 * 2 - 0.01 * 0.5)
    assert trainer.training_step == 1


def test_curriculum_anneals_lambda1(trainer: REINFORCETrainer) -> None:
    """λ₁ should slide from 0.5 toward 0.1 across 50K steps."""
    trainer.training_step = 0
    early = trainer.train_step([], [], 0.0, 0, 0.0)
    trainer.training_step = 50_000
    late = trainer.train_step([], [], 0.0, 0, 0.0)
    assert early["lambda1"] > late["lambda1"]
    assert late["lambda1"] == pytest.approx(0.1, abs=1e-3)

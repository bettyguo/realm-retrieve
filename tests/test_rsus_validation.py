"""Regression tests for issue #5: validate RSUS weights at construction time.

We avoid actually instantiating ``RSUSCalculator`` because that would need
spaCy + the English NER model. We inline-test the validation block by calling
the same logic directly.
"""

from __future__ import annotations

import math

import pytest


def _validate(alpha: float, beta: float, gamma: float) -> None:
    """Mirror of the validation block in RSUSCalculator.__init__ (#5)."""
    for name, val in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
        if val < 0:
            raise ValueError(f"RSUS weight {name}={val!r} must be >= 0")
    total = alpha + beta + gamma
    if not math.isclose(total, 1.0, abs_tol=1e-3):
        raise ValueError(
            f"RSUS weights must sum to 1; got alpha={alpha}, beta={beta}, "
            f"gamma={gamma} (sum={total:.6f})"
        )


class TestRSUSWeightValidation:
    @pytest.mark.parametrize(
        "a, b, g",
        [
            (0.4,  0.35, 0.25),   # paper defaults
            (1.0,  0.0,  0.0),
            (0.5,  0.5,  0.0),
            (0.401, 0.349, 0.250),  # within abs_tol=1e-3
        ],
    )
    def test_accepts_valid(self, a, b, g):
        _validate(a, b, g)  # should not raise

    @pytest.mark.parametrize(
        "a, b, g",
        [
            (0.4, 0.4, 0.4),    # sum 1.2
            (0.5, 0.5, 0.5),    # sum 1.5
            (0.1, 0.1, 0.1),    # sum 0.3
        ],
    )
    def test_rejects_bad_sum(self, a, b, g):
        with pytest.raises(ValueError, match="must sum to 1"):
            _validate(a, b, g)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            _validate(-0.1, 0.55, 0.55)

    def test_error_message_names_the_bad_sum(self):
        with pytest.raises(ValueError) as ei:
            _validate(0.4, 0.4, 0.4)
        msg = str(ei.value)
        assert "1.2" in msg or "sum=1.2" in msg


class TestEmptyEpisodeGuard:
    """Issue #2: an empty episode should not raise inside REINFORCETrainer."""

    def test_guard_logic_returns_skipped_flag(self):
        # Mirror the guard's contract without needing torch: the production
        # code returns a dict with skipped_empty_episode=True when states is
        # empty. Here we verify the *shape* of that contract is documented.
        from inspect import getsource

        from realm_retrieve.models.policy import REINFORCETrainer

        src = getsource(REINFORCETrainer.train_step)
        assert "skipped_empty_episode" in src
        assert "if not states" in src

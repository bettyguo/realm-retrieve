"""Tests for ``RSUSCalculator`` construction (Issue #5).

We deliberately avoid loading spaCy or hitting the LRM — those paths are
covered by the integration tests. Here we just lock in the constructor
contract.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from realm_retrieve.models.rsus import RSUSCalculator


def _construct(alpha: float, beta: float, gamma: float) -> RSUSCalculator:
    """Construct a calculator while stubbing out the spaCy load."""
    with patch("spacy.load") as load:
        load.return_value = object()
        return RSUSCalculator(
            reasoning_model=None,
            retriever=None,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )


def test_weights_sum_to_one_is_accepted() -> None:
    calc = _construct(0.4, 0.35, 0.25)
    assert calc.alpha == pytest.approx(0.4)
    assert calc.beta == pytest.approx(0.35)
    assert calc.gamma == pytest.approx(0.25)


@pytest.mark.parametrize(
    "alpha, beta, gamma",
    [
        (0.4, 0.4, 0.4),    # sum 1.20
        (0.2, 0.2, 0.2),    # sum 0.60
        (0.5, 0.5, 0.5),    # sum 1.50
    ],
)
def test_weights_must_sum_to_one(alpha: float, beta: float, gamma: float) -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        _construct(alpha, beta, gamma)


def test_negative_weight_rejected() -> None:
    with pytest.raises(ValueError, match="must be >= 0|non-negative"):
        _construct(-0.1, 0.6, 0.5)


def test_tolerates_tiny_rounding_error() -> None:
    # 0.333… recurring shouldn't trip the validator.
    _construct(1 / 3, 1 / 3, 1 / 3)

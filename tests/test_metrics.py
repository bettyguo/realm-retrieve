"""Unit tests for :mod:`realm_retrieve.evaluation.metrics`.

These exercise the pure-Python paths (no GPU / network / large checkpoints).
"""

from __future__ import annotations

import numpy as np
import pytest

from realm_retrieve.evaluation.metrics import (
    compute_efficiency_metrics,
    compute_exact_match,
    compute_f1,
    compute_qa_metrics,
    normalize_answer,
    paired_bootstrap_test,
)


class TestNormalizeAnswer:
    def test_lowercases(self):
        assert normalize_answer("BERLIN") == "berlin"

    def test_strips_articles(self):
        assert normalize_answer("The Berlin Wall") == "berlin wall"

    def test_strips_punctuation(self):
        assert normalize_answer("Berlin, 1989.") == "berlin 1989"

    def test_collapses_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"


class TestExactMatch:
    @pytest.mark.parametrize(
        "pred, gold, expected",
        [
            ("Beijing", "beijing", 1.0),
            ("The Beijing", "Beijing", 1.0),
            ("Tokyo", "Beijing", 0.0),
            ("Beijing.", "Beijing", 1.0),
        ],
    )
    def test_em(self, pred, gold, expected):
        assert compute_exact_match(pred, gold) == expected


class TestF1:
    def test_identical_is_one(self):
        assert compute_f1("Alan Turing", "Alan Turing") == pytest.approx(1.0)

    def test_disjoint_is_zero(self):
        assert compute_f1("Tokyo", "Beijing") == 0.0

    def test_partial_overlap(self):
        # pred="Berlin Wall fell", gold="The Berlin Wall"
        # pred tokens: berlin, wall, fell  (3)
        # gold tokens: berlin, wall         (2)
        # common = 2 → p = 2/3, r = 2/2 = 1, F1 = 2*(2/3)/(2/3+1) = 0.8
        assert compute_f1("Berlin Wall fell", "The Berlin Wall") == pytest.approx(0.8)

    def test_empty_prediction(self):
        assert compute_f1("", "Berlin") == 0.0


class TestQAMetrics:
    def test_perfect_run(self):
        out = compute_qa_metrics(["Beijing", "Tokyo"], ["Beijing", "Tokyo"])
        assert out["em"] == 100.0
        assert out["f1"] == 100.0

    def test_mixed(self):
        out = compute_qa_metrics(["Beijing", "Tokyo"], ["Beijing", "Kyoto"])
        assert out["em"] == 50.0
        assert 0 < out["f1"] < 100


class TestEfficiencyMetrics:
    def test_basic(self):
        out = compute_efficiency_metrics(
            retrieval_calls=[1, 2, 3],
            latencies=[0.1, 0.2, 0.3],
            reasoning_tokens=[100, 200, 300],
        )
        assert out["avg_retrieval_calls"] == pytest.approx(2.0)
        assert out["total_retrieval_calls"] == 6
        assert out["avg_latency"] == pytest.approx(0.2)
        assert out["per_call_overhead"] == pytest.approx(0.1)

    def test_zero_calls(self):
        out = compute_efficiency_metrics([0, 0], [0.0, 0.0], [10, 20])
        assert out["per_call_overhead"] == 0.0


class TestBootstrap:
    def test_identical_distributions_centered_at_zero(self):
        rng = np.random.default_rng(0)
        s = rng.normal(size=200).tolist()
        diff, (lo, hi), p = paired_bootstrap_test(s, s, num_iterations=500)
        assert diff == 0.0
        assert lo == 0.0 and hi == 0.0
        # p-value: P(|D| >= 0) is always 1 by definition
        assert p == 1.0

    def test_clear_separation_has_ci_excluding_zero(self):
        rng = np.random.default_rng(1)
        better = rng.normal(loc=0.8, scale=0.05, size=100).tolist()
        worse = rng.normal(loc=0.5, scale=0.05, size=100).tolist()
        diff, (lo, hi), _ = paired_bootstrap_test(better, worse, num_iterations=1000)
        assert diff > 0.2
        # The 95% CI of the difference excludes zero — the real significance signal.
        assert lo > 0.0
        assert hi > lo

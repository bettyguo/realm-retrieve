#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Calibrate RSUS weights, retrieval threshold, and compression threshold.

Performs three calibration sweeps against a dev set:
  1. Weight calibration  -- grid search over (alpha, beta, gamma)
  2. Threshold calibration -- sweep tau for retrieval decisions
  3. Compression calibration -- sweep tau_rel for content pruning

Usage:
    python scripts/training/calibrate_rsus.py \
        --data data/dev_segments.jsonl \
        --segmenter_checkpoint checkpoints/segmenter \
        --output results/rsus_calibration \
        --mode all
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project root / import path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from realm_retrieve.models.rsus import RSUSCalculator, RSUSComponents  # noqa: E402
from realm_retrieve.models.segmentation import (  # noqa: E402
    ReasoningStepSegmenter,
    StepBoundaryClassifier,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dev_data(path: str) -> List[Dict[str, Any]]:
    """Load dev-set JSONL.

    Each record is expected to have:
        - "text": reasoning step text
        - "context": prior reasoning context
        - "retrieval_benefit": float score from downstream evaluation
        - "gold_retrieval": 0 or 1 binary label
        - "relevance_score": float relevance of the segment (for compression)
    Missing fields are filled with defaults so the calibration
    pipeline can still run for smoke-testing.
    """
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("Loaded %d records from %s", len(records), path)
    return records


def _ensure_fields(
    records: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Validate that required fields are present in each record."""
    required = ["retrieval_benefit", "gold_retrieval", "relevance_score",
                 "u_verb", "u_ent", "u_cons"]
    for rec in records:
        if "text" not in rec:
            rec["text"] = rec.get("step_text", "")
        if "context" not in rec:
            rec["context"] = rec.get("prior_context", "")
        missing = [f for f in required if f not in rec]
        if missing:
            raise ValueError(
                f"Record missing required fields: {missing}. "
                f"Run RSUS signal extraction before calibration."
            )
    return records


# ---------------------------------------------------------------------------
# RSUS score computation helpers
# ---------------------------------------------------------------------------

def compute_rsus_scores(
    records: List[Dict[str, Any]],
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Compute RSUS = alpha*u_verb + beta*u_ent + gamma*u_cons for each record."""
    scores = np.array(
        [
            alpha * rec["u_verb"] + beta * rec["u_ent"] + gamma * rec["u_cons"]
            for rec in records
        ],
        dtype=np.float64,
    )
    return scores


# ---------------------------------------------------------------------------
# 1. Weight calibration
# ---------------------------------------------------------------------------

def calibrate_weights(
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Grid search over alpha, beta, gamma with constraint alpha+beta+gamma=1.

    Returns the best weight triple and its Pearson correlation with the
    downstream retrieval benefit signal.
    """
    from scipy.stats import pearsonr

    benefits = np.array([rec["retrieval_benefit"] for rec in records], dtype=np.float64)

    step = 0.05
    grid_values = [round(v * step, 2) for v in range(int(1.0 / step) + 1)]

    results: List[Dict[str, Any]] = []
    total_combos = 0

    # Count valid combos for progress reporting.
    for a in grid_values:
        for b in grid_values:
            g = round(1.0 - a - b, 10)
            if 0.0 <= g <= 1.0 and math.isclose(a + b + g, 1.0, abs_tol=1e-9):
                total_combos += 1

    log.info(
        "Weight calibration: %d valid (alpha, beta, gamma) combinations", total_combos
    )

    done = 0
    for a in grid_values:
        for b in grid_values:
            g = round(1.0 - a - b, 10)
            if g < -1e-9 or g > 1.0 + 1e-9:
                continue
            if not math.isclose(a + b + g, 1.0, abs_tol=1e-9):
                continue
            g = max(0.0, min(1.0, g))  # clamp floating-point residual

            scores = compute_rsus_scores(records, a, b, g)

            # Pearson correlation (handle degenerate cases)
            if np.std(scores) < 1e-12 or np.std(benefits) < 1e-12:
                corr = 0.0
            else:
                corr, _ = pearsonr(scores, benefits)
                if np.isnan(corr):
                    corr = 0.0

            results.append(
                {
                    "alpha": round(a, 2),
                    "beta": round(b, 2),
                    "gamma": round(g, 2),
                    "correlation": round(corr, 6),
                }
            )

            done += 1
            if done % 50 == 0 or done == total_combos:
                print(
                    f"  Weight grid progress: {done}/{total_combos} "
                    f"({100 * done / total_combos:.0f}%)"
                )

    # Sort descending by correlation, keep top 20.
    results.sort(key=lambda r: r["correlation"], reverse=True)
    top_20 = results[:20]

    out_path = output_dir / "rsus_weight_grid.json"
    with open(out_path, "w") as f:
        json.dump(top_20, f, indent=2)
    log.info("Weight grid results saved to %s", out_path)

    best = top_20[0]
    print(f"\n{'=' * 60}")
    print("  Weight Calibration Results")
    print(f"{'=' * 60}")
    print(
        f"  Best weights:  alpha={best['alpha']:.2f}  "
        f"beta={best['beta']:.2f}  gamma={best['gamma']:.2f}"
    )
    print(f"  Correlation:   {best['correlation']:.4f}")

    # Report equal-weights baseline for comparison.
    equal = next(
        (
            r
            for r in results
            if math.isclose(r["alpha"], 1 / 3, abs_tol=0.02)
            and math.isclose(r["beta"], 1 / 3, abs_tol=0.02)
            and math.isclose(r["gamma"], 1 / 3, abs_tol=0.02)
        ),
        None,
    )
    if equal is not None:
        print(
            f"  Equal weights: alpha=0.33  beta=0.33  gamma=0.33  "
            f"corr={equal['correlation']:.4f}"
        )
    print(f"{'=' * 60}\n")

    return best


# ---------------------------------------------------------------------------
# 2. Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_threshold(
    records: List[Dict[str, Any]],
    output_dir: Path,
    alpha: float = 0.4,
    beta: float = 0.35,
    gamma: float = 0.25,
) -> Dict[str, Any]:
    """Sweep retrieval threshold tau and compute precision / recall / F1.

    Uses the given (alpha, beta, gamma) weights to compute RSUS scores, then
    binarises with threshold tau to produce retrieval decisions.
    """
    scores = compute_rsus_scores(records, alpha, beta, gamma)
    gold = np.array([rec["gold_retrieval"] for rec in records], dtype=np.int32)

    step = 0.05
    thresholds = [round(0.1 + i * step, 2) for i in range(int((0.9 - 0.1) / step) + 1)]

    results: List[Dict[str, Any]] = []

    for tau in thresholds:
        predicted = (scores >= tau).astype(np.int32)

        tp = int(np.sum((predicted == 1) & (gold == 1)))
        fp = int(np.sum((predicted == 1) & (gold == 0)))
        fn = int(np.sum((predicted == 0) & (gold == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results.append(
            {
                "threshold": round(tau, 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )

    # Sort by F1 descending.
    results.sort(key=lambda r: r["f1"], reverse=True)

    out_path = output_dir / "rsus_threshold_calibration.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Threshold calibration saved to %s", out_path)

    best = results[0]
    print(f"\n{'=' * 60}")
    print("  Threshold Calibration Results")
    print(f"{'=' * 60}")
    print(f"  Best threshold:  tau={best['threshold']:.2f}")
    print(f"  Precision:       {best['precision']:.4f}")
    print(f"  Recall:          {best['recall']:.4f}")
    print(f"  F1:              {best['f1']:.4f}")
    print(f"{'=' * 60}\n")

    return best


# ---------------------------------------------------------------------------
# 3. Compression calibration
# ---------------------------------------------------------------------------

def calibrate_compression(
    records: List[Dict[str, Any]],
    output_dir: Path,
    alpha: float = 0.4,
    beta: float = 0.35,
    gamma: float = 0.25,
) -> Dict[str, Any]:
    """Sweep compression threshold tau_rel and compute compression / utility.

    Segments with RSUS relevance scores **below** tau_rel are pruned.
    """
    scores = compute_rsus_scores(records, alpha, beta, gamma)
    gold = np.array([rec["gold_retrieval"] for rec in records], dtype=np.int32)
    relevance = np.array(
        [rec["relevance_score"] for rec in records], dtype=np.float64,
    )

    step = 0.05
    thresholds = [round(0.1 + i * step, 2) for i in range(int((0.9 - 0.1) / step) + 1)]

    results: List[Dict[str, Any]] = []

    for tau_rel in thresholds:
        retained_mask = relevance >= tau_rel

        n_total = len(records)
        n_retained = int(np.sum(retained_mask))
        compression_ratio = 1.0 - n_retained / n_total if n_total > 0 else 0.0
        content_retained = n_retained / n_total if n_total > 0 else 0.0

        # Utility preserved: fraction of gold-positive retrievals that are
        # still triggered among the retained segments.
        gold_positive_mask = gold == 1
        useful_retained = int(np.sum(retained_mask & gold_positive_mask))
        total_useful = int(np.sum(gold_positive_mask))
        utility_preserved = (
            useful_retained / total_useful if total_useful > 0 else 1.0
        )

        results.append(
            {
                "threshold": round(tau_rel, 2),
                "compression_ratio": round(compression_ratio, 4),
                "content_retained": round(content_retained, 4),
                "utility_preserved": round(utility_preserved, 4),
            }
        )

    # Sort by a combined objective: high compression + high utility.
    results.sort(
        key=lambda r: r["compression_ratio"] + r["utility_preserved"], reverse=True,
    )

    out_path = output_dir / "rsus_compression_calibration.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Compression calibration saved to %s", out_path)

    best = results[0]
    print(f"\n{'=' * 60}")
    print("  Compression Calibration Results")
    print(f"{'=' * 60}")
    print(f"  Best threshold:      tau_rel={best['threshold']:.2f}")
    print(
        f"  Compression ratio:   {best['compression_ratio']:.0%} "
        f"(content removed)"
    )
    print(f"  Content retained:    {best['content_retained']:.0%}")
    print(f"  Utility preserved:   {best['utility_preserved']:.0%}")
    print(f"{'=' * 60}\n")

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate RSUS weights, retrieval threshold, and compression threshold.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Dev data JSONL path.",
    )
    parser.add_argument(
        "--segmenter_checkpoint",
        required=True,
        type=Path,
        help="Path to trained segmenter checkpoint.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for calibration results.",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["weights", "threshold", "compression", "all"],
        help="Calibration mode (default: all).",
    )
    parser.add_argument(
        "--base_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model name for the segmenter (default: sentence-transformers/all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  RSUS Calibration")
    print(f"{'=' * 60}")
    print(f"  Data:        {args.data}")
    print(f"  Checkpoint:  {args.segmenter_checkpoint}")
    print(f"  Output:      {args.output}")
    print(f"  Mode:        {args.mode}")
    print(f"  Base model:  {args.base_model}")
    print(f"  Seed:        {args.seed}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    records = load_dev_data(str(args.data))
    records = _ensure_fields(records, rng)

    # ------------------------------------------------------------------
    # Optionally load segmenter (used only if raw traces need segmenting)
    # ------------------------------------------------------------------
    segmenter_ckpt = args.segmenter_checkpoint
    if segmenter_ckpt.exists():
        log.info("Loading segmenter from %s", segmenter_ckpt)
        try:
            model = StepBoundaryClassifier(base_model=args.base_model)
            import torch
            state = torch.load(
                segmenter_ckpt / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state)
            log.info("Segmenter loaded successfully")
        except Exception as exc:
            log.warning("Could not load segmenter checkpoint: %s", exc)
    else:
        log.info(
            "Segmenter checkpoint not found at %s; proceeding with "
            "pre-computed segments from the data file.",
            segmenter_ckpt,
        )

    # ------------------------------------------------------------------
    # Run calibrations
    # ------------------------------------------------------------------
    best_alpha, best_beta, best_gamma = 0.4, 0.35, 0.25

    if args.mode in ("weights", "all"):
        best_w = calibrate_weights(records, args.output)
        best_alpha = best_w["alpha"]
        best_beta = best_w["beta"]
        best_gamma = best_w["gamma"]

    if args.mode in ("threshold", "all"):
        calibrate_threshold(
            records, args.output,
            alpha=best_alpha, beta=best_beta, gamma=best_gamma,
        )

    if args.mode in ("compression", "all"):
        calibrate_compression(
            records, args.output,
            alpha=best_alpha, beta=best_beta, gamma=best_gamma,
        )

    print("Done -- results saved to", args.output)


if __name__ == "__main__":
    main()

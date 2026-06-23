#!/usr/bin/env python3
"""Aggregate per-seed prediction files and produce Table 1.

Reads JSONL prediction files for each (method, dataset, model, seed)
combination, computes per-seed EM and F1 by recomputing metrics from
stored predicted/gold answers, averages across seeds, and prints a
reproducible Table 1 with deltas versus the no_retrieval baseline.

All percentage metrics use Python's built-in round() to 1 decimal place
(banker's rounding / round-half-to-even), applied once at the final
reporting stage — never to intermediate values.

Usage:
    python scripts/evaluation/aggregate_seeds.py \
        --predictions_dir predictions/ \
        --seeds 42 43 44

    python scripts/evaluation/aggregate_seeds.py \
        --predictions_dir outputs/calibrated_results/ \
        --seeds 42 43 44 \
        --methods realm_retrieve no_retrieval single_rag ircot flare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from realm_retrieve.evaluation.metrics import compute_exact_match, compute_f1


ALL_METHODS = [
    "realm_retrieve",
    "no_retrieval",
    "single_rag",
    "ircot",
    "flare",
    "self_rag",
    "search_r1",
    "naive_interleave",
]

ALL_DATASETS = ["musique", "hotpotqa", "2wikimhqa"]

MODEL_ALIASES = {
    "r1_32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "r1_671b": "deepseek-ai/DeepSeek-R1",
    "qwq_32b": "Qwen/QwQ-32B-Preview",
}


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def find_prediction_file(
    predictions_dir: Path, method: str, dataset: str, model: str, seed: int,
) -> Path | None:
    slug = model_slug(model)
    candidates = [
        predictions_dir / f"{method}_{dataset}_{slug}_seed{seed}.jsonl",
        predictions_dir / f"{method}_{dataset}_{model}_seed{seed}.jsonl",
    ]
    for alias, full_name in MODEL_ALIASES.items():
        if model == full_name:
            candidates.append(
                predictions_dir / f"{method}_{dataset}_{alias}_seed{seed}.jsonl"
            )
    for c in candidates:
        if c.exists():
            return c
    return None


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_seed_metrics(records: list[dict]) -> dict[str, float]:
    """Recompute EM and F1 from stored answers — returns raw (unrounded) values."""
    em_scores = []
    f1_scores = []
    for rec in records:
        pred = rec.get("predicted_answer", "")
        gold = rec.get("gold_answer", "")
        em_scores.append(compute_exact_match(pred, gold))
        f1_scores.append(compute_f1(pred, gold))
    return {
        "em": float(np.mean(em_scores)) * 100,
        "f1": float(np.mean(f1_scores)) * 100,
        "n": len(records),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-seed predictions into Table 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions_dir", type=str, required=True,
        help="Directory containing per-seed JSONL prediction files",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Seeds to aggregate",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=None,
        help="Methods to include (default: all found)",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="Datasets to include (default: all found)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Models to include (default: all found)",
    )
    parser.add_argument(
        "--baseline", type=str, default="no_retrieval",
        help="Baseline method for computing deltas",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Optional path to write full results as JSON",
    )
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    if not predictions_dir.is_dir():
        print(f"Error: {predictions_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Discover available files
    methods = set(args.methods) if args.methods else set(ALL_METHODS)
    datasets = set(args.datasets) if args.datasets else set(ALL_DATASETS)
    models_to_check = args.models or list(MODEL_ALIASES.keys()) + list(MODEL_ALIASES.values())

    # Collect per-seed metrics
    # results[dataset][model][method] = {"em": [...per seed], "f1": [...per seed]}
    results: dict[str, dict[str, dict[str, dict]]] = {}

    for dataset in sorted(datasets):
        for model in models_to_check:
            for method in sorted(methods):
                seed_ems = []
                seed_f1s = []
                for seed in args.seeds:
                    path = find_prediction_file(predictions_dir, method, dataset, model, seed)
                    if path is None:
                        continue
                    records = load_records(path)
                    if not records:
                        continue
                    metrics = compute_seed_metrics(records)
                    seed_ems.append(metrics["em"])
                    seed_f1s.append(metrics["f1"])

                if not seed_ems:
                    continue

                canonical_model = model
                for alias, full_name in MODEL_ALIASES.items():
                    if model == full_name:
                        canonical_model = alias
                        break

                results.setdefault(dataset, {}).setdefault(canonical_model, {})[method] = {
                    "per_seed_em": seed_ems,
                    "per_seed_f1": seed_f1s,
                    "mean_em": float(np.mean(seed_ems)),
                    "mean_f1": float(np.mean(seed_f1s)),
                    "num_seeds": len(seed_ems),
                }

    if not results:
        print("No prediction files found.", file=sys.stderr)
        sys.exit(1)

    # Print table
    print()
    print("=" * 80)
    print("Table 1: Multi-hop QA results (3-seed average)")
    print("=" * 80)
    print(f"  Rounding: Python round() to 1 decimal (round-half-to-even)")
    print(f"  Seeds: {args.seeds}")
    print(f"  Baseline for deltas: {args.baseline}")
    print()

    header = f"{'Setting':<30} {'Method':<20} {'EM':>6} {'F1':>6} {'ΔEM':>6} {'ΔF1':>6} {'Seeds':>5}"
    print(header)
    print("-" * len(header))

    for dataset in sorted(results.keys()):
        for model in sorted(results[dataset].keys()):
            model_results = results[dataset][model]
            baseline = model_results.get(args.baseline)
            baseline_em = baseline["mean_em"] if baseline else 0.0
            baseline_f1 = baseline["mean_f1"] if baseline else 0.0

            setting = f"{dataset}, {model}"

            for method in sorted(model_results.keys()):
                m = model_results[method]
                em = round(m["mean_em"], 1)
                f1 = round(m["mean_f1"], 1)

                if method == args.baseline:
                    delta_em_str = "  ---"
                    delta_f1_str = "  ---"
                else:
                    raw_delta_em = m["mean_em"] - baseline_em
                    raw_delta_f1 = m["mean_f1"] - baseline_f1
                    delta_em = round(raw_delta_em, 1)
                    delta_f1 = round(raw_delta_f1, 1)
                    delta_em_str = f"{delta_em:+.1f}"
                    delta_f1_str = f"{delta_f1:+.1f}"

                print(
                    f"{setting:<30} {method:<20} {em:>6.1f} {f1:>6.1f} "
                    f"{delta_em_str:>6} {delta_f1_str:>6} {m['num_seeds']:>5}"
                )
                setting = ""  # only print once per group

            print()

    # Print raw (unrounded) values for transparency
    print("=" * 80)
    print("Raw (unrounded) values for reproducibility verification")
    print("=" * 80)
    print()

    for dataset in sorted(results.keys()):
        for model in sorted(results[dataset].keys()):
            model_results = results[dataset][model]
            baseline = model_results.get(args.baseline)
            baseline_em = baseline["mean_em"] if baseline else 0.0
            baseline_f1 = baseline["mean_f1"] if baseline else 0.0

            print(f"{dataset}, {model}:")
            for method in sorted(model_results.keys()):
                m = model_results[method]
                delta_em = m["mean_em"] - baseline_em
                delta_f1 = m["mean_f1"] - baseline_f1
                print(
                    f"  {method:<20} EM={m['mean_em']:.4f}  F1={m['mean_f1']:.4f}  "
                    f"ΔEM={delta_em:+.4f}  ΔF1={delta_f1:+.4f}  "
                    f"per-seed EM={m['per_seed_em']}  per-seed F1={m['per_seed_f1']}"
                )
            print()

    # Write JSON output
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()

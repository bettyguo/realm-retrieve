#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Multi-Seed Experiment Runner for ReaLM-Retrieve.

Launches training for each seed sequentially (or in parallel), then
aggregates results and validates against paper-reported mean +/- std.

Usage:
    # Train segmenter with 3 seeds
    python scripts/training/multi_seed_runner.py \
        --component segmenter \
        --seeds 42,123,456 \
        --base_config configs/experiments/train_segmentation.yaml \
        --output checkpoints/

    # Train policy with 3 seeds
    python scripts/training/multi_seed_runner.py \
        --component policy \
        --seeds 42,123,456 \
        --base_config configs/experiments/train_policy.yaml \
        --output checkpoints/

    # Aggregate results
    python scripts/training/multi_seed_runner.py \
        --component policy --mode aggregate \
        --seeds 42,123,456 \
        --log_dir outputs/training_logs/ \
        --output outputs/training_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


COMPONENT_SCRIPTS = {
    "segmenter": "scripts/training/train_segmenter_full.py",
    "policy": "scripts/training/train_policy_full.py",
    "proxy_mlp": "scripts/training/train_proxy_mlp.py",
}

COMPONENT_METRIC = {
    "segmenter": "test_f1",
    "policy": "dev_f1",
    "proxy_mlp": "dev_auroc",
}

PAPER_EXPECTED: Dict[str, Dict] = {}


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        print("WARNING: PyYAML not installed, config loading skipped")
        return {}


def build_train_command(
    component: str,
    seed: int,
    config: Dict,
    output_dir: str,
    log_file: str,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Build the training command for a given component and seed."""
    script = COMPONENT_SCRIPTS[component]
    cmd = [sys.executable, script, "--seed", str(seed)]

    if component == "segmenter":
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        train_cfg = config.get("training", {})

        cmd.extend([
            "--data", data_cfg.get("train_path", "data/processed/segmentation/train.jsonl"),
            "--test_data", data_cfg.get("val_path", "data/processed/segmentation/val.jsonl"),
            "--hidden_dim", str(model_cfg.get("hidden_dim", 256)),
            "--n_layers", str(model_cfg.get("num_layers", 3)),
            "--n_heads", str(model_cfg.get("num_heads", 4)),
            "--epochs", str(train_cfg.get("num_epochs", 10)),
            "--lr", str(train_cfg.get("learning_rate", 5e-5)),
            "--batch_size", str(train_cfg.get("batch_size", 32)),
            "--output", output_dir,
            "--log_file", log_file,
        ])

    elif component == "policy":
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        train_cfg = config.get("training", {})

        cmd.extend([
            "--train_data", data_cfg.get("train_path", "data/processed/musique/train.jsonl"),
            "--dev_data", data_cfg.get("val_path", "data/processed/musique/dev.jsonl"),
            "--policy_hidden_dim", str(model_cfg.get("hidden_dim", 512)),
            "--policy_layers", str(model_cfg.get("num_layers", 4)),
            "--policy_heads", str(model_cfg.get("num_heads", 8)),
            "--steps", str(train_cfg.get("num_steps", 50000)),
            "--lr", str(train_cfg.get("learning_rate", 1e-4)),
            "--batch_size", str(train_cfg.get("batch_size", 64)),
            "--lambda1_start", str(train_cfg.get("lambda1_start", 0.5)),
            "--lambda1_end", str(train_cfg.get("lambda1_end", 0.1)),
            "--save_every", str(train_cfg.get("save_every", 5000)),
            "--output", output_dir,
            "--log_file", log_file,
        ])

    elif component == "proxy_mlp":
        cmd.extend([
            "--data", config.get("data", {}).get("train_path", "data/annotations/nq_train_traces.jsonl"),
            "--output", output_dir,
            "--log_file", log_file,
            "--epochs", str(config.get("training", {}).get("epochs", 50)),
            "--lr", str(config.get("training", {}).get("lr", 1e-3)),
        ])
        test_path = config.get("data", {}).get("test_path")
        if test_path:
            cmd.extend(["--test_data", test_path])

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def run_single_seed(
    component: str,
    seed: int,
    config: Dict,
    base_output: str,
    base_log_dir: str,
    extra_args: Optional[List[str]] = None,
) -> Dict:
    """Run training for a single seed and return result summary."""
    output_dir = os.path.join(base_output, component, f"seed_{seed}")
    log_file = os.path.join(base_log_dir, f"{component}_seed{seed}.jsonl")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    cmd = build_train_command(component, seed, config, output_dir, log_file, extra_args)
    print(f"\n{'='*60}")
    print(f"  Running {component} with seed={seed}")
    print(f"  Output: {output_dir}")
    print(f"  Log:    {log_file}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Training failed for seed {seed}")
        print(f"STDERR: {result.stderr[:2000]}")
        return {"seed": seed, "status": "failed", "error": result.stderr[:500]}

    final_metrics = extract_final_metrics(log_file, component)
    final_metrics["seed"] = seed
    final_metrics["status"] = "completed"
    final_metrics["output_dir"] = output_dir
    final_metrics["log_file"] = log_file

    print(f"Seed {seed} completed: {final_metrics}")
    return final_metrics


def extract_final_metrics(log_file: str, component: str) -> Dict:
    """Extract final metrics from a training log JSONL file."""
    if not os.path.exists(log_file):
        return {}

    last_entry = None
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                last_entry = json.loads(line)

    if last_entry is None:
        return {}

    metric_key = COMPONENT_METRIC.get(component, "metric_value")
    return {
        metric_key: last_entry.get(metric_key, last_entry.get("metric_value")),
        "final_loss": last_entry.get("train_loss", last_entry.get("loss")),
    }


def aggregate_results(
    component: str,
    seeds: List[int],
    log_dir: str,
    output_path: str,
) -> Dict:
    """Aggregate results across seeds and validate against paper numbers."""
    all_metrics: Dict[str, List[float]] = {}
    seed_results = []

    for seed in seeds:
        log_file = os.path.join(log_dir, f"{component}_seed{seed}.jsonl")
        if not os.path.exists(log_file):
            print(f"WARNING: Log file not found for seed {seed}: {log_file}")
            continue

        metrics = extract_final_metrics(log_file, component)
        metrics["seed"] = seed
        seed_results.append(metrics)

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                all_metrics.setdefault(key, []).append(value)

    summary = {
        "component": component,
        "n_seeds": len(seed_results),
        "seeds": seeds,
        "seed_results": seed_results,
        "aggregated": {},
        "validation": [],
    }

    for key, values in all_metrics.items():
        arr = np.array(values)
        summary["aggregated"][key] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "values": [round(float(v), 4) for v in values],
        }

    expected = PAPER_EXPECTED.get(component, {})
    for metric_key, expected_vals in expected.items():
        if metric_key in summary["aggregated"]:
            actual_mean = summary["aggregated"][metric_key]["mean"]
            actual_std = summary["aggregated"][metric_key]["std"]
            expected_mean = expected_vals["mean"]
            expected_std = expected_vals["std"]

            mean_diff = abs(actual_mean - expected_mean)
            within_2sigma = mean_diff <= 2 * expected_std

            check = {
                "metric": metric_key,
                "expected_mean": expected_mean,
                "expected_std": expected_std,
                "actual_mean": actual_mean,
                "actual_std": actual_std,
                "mean_diff": round(mean_diff, 4),
                "within_2sigma": within_2sigma,
                "status": "PASS" if within_2sigma else "FAIL",
            }
            summary["validation"].append(check)

            status = "PASS" if within_2sigma else "FAIL"
            print(f"  [{status}] {metric_key}: expected {expected_mean}±{expected_std}, "
                  f"got {actual_mean}±{actual_std} (diff={mean_diff:.4f})")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {output_path}")

    return summary


def run_parallel_seeds(
    component: str,
    seeds: List[int],
    config: Dict,
    base_output: str,
    base_log_dir: str,
    extra_args: Optional[List[str]] = None,
) -> List[Dict]:
    """Run training for all seeds in parallel using subprocess."""
    processes = []
    for seed in seeds:
        output_dir = os.path.join(base_output, component, f"seed_{seed}")
        log_file = os.path.join(base_log_dir, f"{component}_seed{seed}.jsonl")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        cmd = build_train_command(component, seed, config, output_dir, log_file, extra_args)
        print(f"Launching seed {seed}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append((seed, proc, log_file))

    results = []
    for seed, proc, log_file in processes:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(f"ERROR: Seed {seed} failed: {stderr[:500]}")
            results.append({"seed": seed, "status": "failed", "error": stderr[:500]})
        else:
            metrics = extract_final_metrics(log_file, component)
            metrics["seed"] = seed
            metrics["status"] = "completed"
            results.append(metrics)
            print(f"Seed {seed} completed: {metrics}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Seed Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--component", required=True,
                        choices=["segmenter", "policy", "proxy_mlp"],
                        help="Which component to train")
    parser.add_argument("--seeds", required=True,
                        help="Comma-separated list of seeds (e.g., 42,123,456)")
    parser.add_argument("--base_config", default=None,
                        help="Path to base YAML config file")
    parser.add_argument("--output", default="checkpoints/",
                        help="Base output directory for checkpoints")
    parser.add_argument("--log_dir", default="outputs/training_logs/",
                        help="Directory for training log files")
    parser.add_argument("--mode", default="train",
                        choices=["train", "aggregate"],
                        help="Run mode: train all seeds or aggregate existing logs")
    parser.add_argument("--parallel", action="store_true",
                        help="Run seeds in parallel (default: sequential)")
    parser.add_argument("--extra_args", nargs="*", default=None,
                        help="Extra arguments to pass to the training script")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"\nMulti-Seed Runner: component={args.component}, seeds={seeds}, mode={args.mode}")

    if args.mode == "aggregate":
        output_path = args.output if args.output.endswith(".json") else \
            os.path.join(args.output, f"{args.component}_summary.json")
        aggregate_results(args.component, seeds, args.log_dir, output_path)
        return

    config = load_config(args.base_config) if args.base_config else {}

    if args.parallel:
        results = run_parallel_seeds(
            args.component, seeds, config,
            args.output, args.log_dir, args.extra_args,
        )
    else:
        results = []
        for seed in seeds:
            result = run_single_seed(
                args.component, seed, config,
                args.output, args.log_dir, args.extra_args,
            )
            results.append(result)

    print(f"\n{'='*60}")
    print(f"  All seeds completed for {args.component}")
    print(f"{'='*60}")

    completed = [r for r in results if r.get("status") == "completed"]
    failed = [r for r in results if r.get("status") == "failed"]
    print(f"  Completed: {len(completed)}/{len(seeds)}")
    if failed:
        print(f"  Failed: {[r['seed'] for r in failed]}")

    if completed:
        output_path = os.path.join(args.output, f"{args.component}_summary.json")
        aggregate_results(args.component, seeds, args.log_dir, output_path)


if __name__ == "__main__":
    main()

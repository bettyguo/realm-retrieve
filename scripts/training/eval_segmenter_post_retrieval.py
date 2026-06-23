#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Evaluate segmenter on traces with prior retrieval injections.

Measures how well the step boundary classifier performs when reasoning
traces include injected retrieval context from previous pipeline stages.
Computes degradation in boundary F1, shift in RSUS correlation, and
correlation between segmentation errors and downstream retrieval errors.

Usage:
    python scripts/training/eval_segmenter_post_retrieval.py \
        --checkpoint checkpoints/segmentation \
        --data data/post_retrieval/test_with_retrieval.jsonl \
        --baseline_data data/processed/segmentation/test.jsonl \
        --output results/segmenter_post_retrieval_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from realm_retrieve.models.segmentation import StepBoundaryClassifier


# ---------------------------------------------------------------------------
# Dataset (mirrors SegmentationDataset from train_segmentation.py)
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """Dataset for reasoning step boundary detection."""

    def __init__(self, data_path: str, tokenizer, window_size: int = 128):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.samples = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        samples = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        tokens = self.tokenizer.encode(sample["text"], add_special_tokens=True)
        labels = sample["boundary_labels"]  # [0, 1, 0, 0, 1, ...]

        if len(tokens) < self.window_size:
            padding = [self.tokenizer.pad_token_id] * (self.window_size - len(tokens))
            tokens = tokens + padding
            labels = labels + [0] * (self.window_size - len(labels))
        else:
            tokens = tokens[: self.window_size]
            labels = labels[: self.window_size]

        return {
            "input_ids": torch.tensor(tokens),
            "labels": torch.tensor(labels),
            "attention_mask": torch.ones(self.window_size),
        }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_boundary_f1(
    model: StepBoundaryClassifier,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Compute boundary precision, recall, F1 over a dataloader."""
    model.eval()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device))

    total_loss = 0.0
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating boundaries"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            logits_flat = logits.view(-1, 2)
            labels_flat = labels.view(-1)

            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()

            preds = torch.argmax(logits_flat, dim=-1)
            tp += ((preds == 1) & (labels_flat == 1)).sum().item()
            fp += ((preds == 1) & (labels_flat == 0)).sum().item()
            fn += ((preds == 0) & (labels_flat == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loss": total_loss / max(len(dataloader), 1),
    }


def compute_per_trace_boundary_f1(
    model: StepBoundaryClassifier,
    dataset: SegmentationDataset,
    device: str,
    window_size: int,
    stride: int = 64,
) -> List[Dict[str, float]]:
    """Compute per-trace boundary F1 using sliding windows.

    Returns a list (one entry per trace) containing per-trace f1 and
    boundary error rate (1 - f1).
    """
    model.eval()
    per_trace: List[Dict[str, float]] = []

    for idx in tqdm(range(len(dataset)), desc="  Per-trace boundary F1"):
        sample = dataset.samples[idx]
        tokens = dataset.tokenizer.encode(sample["text"], add_special_tokens=True)
        gold_labels = sample["boundary_labels"]
        num_tokens = len(tokens)

        # Sliding window boundary prediction
        boundary_scores = np.zeros(num_tokens)
        boundary_counts = np.zeros(num_tokens)

        for start_idx in range(0, num_tokens, stride):
            end_idx = min(start_idx + window_size, num_tokens)
            window_tokens = tokens[start_idx:end_idx]
            actual_len = len(window_tokens)

            if actual_len < window_size:
                padding = [dataset.tokenizer.pad_token_id] * (window_size - actual_len)
                window_tokens = window_tokens + padding

            input_ids = torch.tensor([window_tokens], device=device)
            attention_mask = torch.ones_like(input_ids)
            attention_mask[0, actual_len:] = 0

            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits[0], dim=-1)[:, 1]

            boundary_scores[start_idx : start_idx + actual_len] += (
                probs[:actual_len].cpu().numpy()
            )
            boundary_counts[start_idx : start_idx + actual_len] += 1

        avg_scores = boundary_scores / np.maximum(boundary_counts, 1)
        preds = (avg_scores > 0.5).astype(int)

        # Truncate gold labels to match token length
        gold = np.array(gold_labels[:num_tokens], dtype=int)
        if len(gold) < num_tokens:
            gold = np.concatenate([gold, np.zeros(num_tokens - len(gold), dtype=int)])

        t_tp = int(((preds == 1) & (gold == 1)).sum())
        t_fp = int(((preds == 1) & (gold == 0)).sum())
        t_fn = int(((preds == 0) & (gold == 1)).sum())

        p = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
        r = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_trace.append({
            "trace_idx": idx,
            "f1": f1,
            "boundary_error": 1.0 - f1,
        })

    return per_trace


def compute_rsus_correlation(
    per_trace_results: List[Dict[str, float]],
    dataset: SegmentationDataset,
) -> Optional[float]:
    """Compute Pearson correlation between per-trace boundary F1 and RSUS.

    Expects each sample to have an 'rsus' field. Returns None if the field
    is not present in the data.
    """
    f1_values = []
    rsus_values = []

    for result in per_trace_results:
        idx = result["trace_idx"]
        sample = dataset.samples[idx]
        rsus = sample.get("rsus")
        if rsus is None:
            continue
        f1_values.append(result["f1"])
        rsus_values.append(float(rsus))

    if len(f1_values) < 3:
        return None

    f1_arr = np.array(f1_values)
    rsus_arr = np.array(rsus_values)

    # Pearson correlation
    corr = float(np.corrcoef(f1_arr, rsus_arr)[0, 1])
    return corr


def compute_seg_retrieval_error_correlation(
    per_trace_results: List[Dict[str, float]],
    dataset: SegmentationDataset,
) -> Optional[float]:
    """Compute Pearson r between segmentation error and retrieval error.

    Expects each sample to have a 'retrieval_error' field. Returns None
    if the field is not present in the data.
    """
    seg_errors = []
    ret_errors = []

    for result in per_trace_results:
        idx = result["trace_idx"]
        sample = dataset.samples[idx]
        ret_err = sample.get("retrieval_error")
        if ret_err is None:
            continue
        seg_errors.append(result["boundary_error"])
        ret_errors.append(float(ret_err))

    if len(seg_errors) < 3:
        return None

    corr = float(np.corrcoef(np.array(seg_errors), np.array(ret_errors))[0, 1])
    return corr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate segmenter on traces with prior retrieval injections"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint directory (contains best_model.pt)",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to test traces JSONL (traces with retrieval context injected)",
    )
    parser.add_argument(
        "--baseline_data",
        required=True,
        help="Path to baseline test traces (without retrieval)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--base_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model name (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--window_size", type=int, default=128, help="Window size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load model from checkpoint
    # ------------------------------------------------------------------
    checkpoint_path = os.path.join(args.checkpoint, "best_model.pt")
    print(f"Loading checkpoint from {checkpoint_path} ...")

    model = StepBoundaryClassifier(
        base_model=args.base_model,
        hidden_dim=args.hidden_dim,
        num_layers=args.n_layers,
        num_heads=args.n_heads,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(
        f"Model loaded (trained epoch {checkpoint.get('epoch', '?')}, "
        f"val F1 {checkpoint.get('f1', '?')})"
    )

    tokenizer = model.tokenizer

    # ------------------------------------------------------------------
    # 1. Evaluate on baseline traces (no retrieval injection)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Baseline evaluation: {args.baseline_data}")
    print("=" * 60)

    baseline_dataset = SegmentationDataset(
        args.baseline_data, tokenizer, window_size=args.window_size
    )
    baseline_loader = DataLoader(
        baseline_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    baseline_metrics = evaluate_boundary_f1(model, baseline_loader, device)
    print(
        f"  Boundary F1: {baseline_metrics['f1']:.4f}  "
        f"(P={baseline_metrics['precision']:.4f}, R={baseline_metrics['recall']:.4f})"
    )

    # Per-trace metrics for correlation analysis
    print("  Computing per-trace metrics for baseline ...")
    baseline_per_trace = compute_per_trace_boundary_f1(
        model, baseline_dataset, device, args.window_size
    )
    rsus_corr_pre = compute_rsus_correlation(baseline_per_trace, baseline_dataset)
    if rsus_corr_pre is not None:
        print(f"  RSUS correlation (pre-retrieval): {rsus_corr_pre:.4f}")
    else:
        print("  RSUS correlation (pre-retrieval): N/A (no rsus field in data)")

    # ------------------------------------------------------------------
    # 2. Evaluate on post-retrieval traces
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Post-retrieval evaluation: {args.data}")
    print("=" * 60)

    post_dataset = SegmentationDataset(
        args.data, tokenizer, window_size=args.window_size
    )
    post_loader = DataLoader(
        post_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    post_metrics = evaluate_boundary_f1(model, post_loader, device)
    print(
        f"  Boundary F1: {post_metrics['f1']:.4f}  "
        f"(P={post_metrics['precision']:.4f}, R={post_metrics['recall']:.4f})"
    )

    # Per-trace metrics for post-retrieval correlation
    print("  Computing per-trace metrics for post-retrieval ...")
    post_per_trace = compute_per_trace_boundary_f1(
        model, post_dataset, device, args.window_size
    )
    rsus_corr_post = compute_rsus_correlation(post_per_trace, post_dataset)
    if rsus_corr_post is not None:
        print(f"  RSUS correlation (post-retrieval): {rsus_corr_post:.4f}")
    else:
        print("  RSUS correlation (post-retrieval): N/A (no rsus field in data)")

    # ------------------------------------------------------------------
    # 3. Segmentation error / retrieval error correlation
    # ------------------------------------------------------------------
    seg_ret_corr = compute_seg_retrieval_error_correlation(post_per_trace, post_dataset)
    if seg_ret_corr is not None:
        print(f"  Seg-error / retrieval-error correlation: {seg_ret_corr:.4f}")
    else:
        print(
            "  Seg-error / retrieval-error correlation: N/A "
            "(no retrieval_error field in data)"
        )

    # ------------------------------------------------------------------
    # 4. Compute deltas
    # ------------------------------------------------------------------
    f1_delta = post_metrics["f1"] - baseline_metrics["f1"]
    rsus_corr_delta = None
    if rsus_corr_pre is not None and rsus_corr_post is not None:
        rsus_corr_delta = rsus_corr_post - rsus_corr_pre

    # ------------------------------------------------------------------
    # 5. Assemble and save results
    # ------------------------------------------------------------------
    results = {
        "pre_retrieval_f1": round(baseline_metrics["f1"], 4),
        "pre_retrieval_precision": round(baseline_metrics["precision"], 4),
        "pre_retrieval_recall": round(baseline_metrics["recall"], 4),
        "post_retrieval_f1": round(post_metrics["f1"], 4),
        "post_retrieval_precision": round(post_metrics["precision"], 4),
        "post_retrieval_recall": round(post_metrics["recall"], 4),
        "f1_delta": round(f1_delta, 4),
        "rsus_correlation_pre": round(rsus_corr_pre, 4) if rsus_corr_pre is not None else None,
        "rsus_correlation_post": round(rsus_corr_post, 4) if rsus_corr_post is not None else None,
        "rsus_correlation_delta": round(rsus_corr_delta, 4) if rsus_corr_delta is not None else None,
        "seg_error_retrieval_error_correlation": (
            round(seg_ret_corr, 4) if seg_ret_corr is not None else None
        ),
        "n_traces_evaluated": len(post_dataset),
        "n_traces_baseline": len(baseline_dataset),
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  Pre-retrieval  F1: {results['pre_retrieval_f1']:.4f}")
    print(f"  Post-retrieval F1: {results['post_retrieval_f1']:.4f}")
    print(f"  F1 delta:          {results['f1_delta']:+.4f}")
    print()
    if results["rsus_correlation_pre"] is not None:
        print(f"  RSUS corr (pre):   {results['rsus_correlation_pre']:.4f}")
        print(f"  RSUS corr (post):  {results['rsus_correlation_post']:.4f}")
        print(f"  RSUS corr delta:   {results['rsus_correlation_delta']:+.4f}")
    else:
        print("  RSUS correlation:  N/A (rsus field not in data)")
    print()
    if results["seg_error_retrieval_error_correlation"] is not None:
        print(
            f"  Seg/retrieval err r: "
            f"{results['seg_error_retrieval_error_correlation']:.4f}"
        )
    else:
        print("  Seg/retrieval err r: N/A (retrieval_error field not in data)")
    print(f"  Traces evaluated:  {results['n_traces_evaluated']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Evaluate trained segmenter on out-of-distribution reasoning traces.

Loads a trained StepBoundaryClassifier from checkpoint and evaluates
boundary detection F1 and average segments per trace across multiple
OOD datasets. Optionally compares against in-domain test performance.

Usage:
    python scripts/training/eval_segmenter_ood.py \
        --checkpoint checkpoints/segmentation \
        --ood_data data/ood/musique.jsonl,data/ood/hotpotqa.jsonl \
        --in_domain_data data/processed/segmentation/test.jsonl \
        --output results/segmenter_ood_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

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

        # Pad or truncate to window_size
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
    """Compute boundary precision, recall, F1 over a dataloader.

    Returns dict with keys: precision, recall, f1, loss.
    """
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


def count_segments_per_trace(
    model: StepBoundaryClassifier,
    dataset: SegmentationDataset,
    device: str,
    window_size: int,
    stride: int = 64,
) -> float:
    """Estimate the average number of predicted segments per trace.

    For each sample, runs sliding-window boundary prediction and counts
    the number of predicted boundary tokens. The segment count equals
    (boundary_count + 1).
    """
    model.eval()
    segment_counts: List[int] = []

    for idx in tqdm(range(len(dataset)), desc="  Counting segments"):
        sample = dataset.samples[idx]
        tokens = dataset.tokenizer.encode(sample["text"], add_special_tokens=True)
        num_tokens = len(tokens)

        boundary_scores = np.zeros(num_tokens)
        boundary_counts = np.zeros(num_tokens)

        for start_idx in range(0, num_tokens, stride):
            end_idx = min(start_idx + window_size, num_tokens)
            window_tokens = tokens[start_idx:end_idx]
            actual_len = len(window_tokens)

            # Pad if needed
            if actual_len < window_size:
                padding = [dataset.tokenizer.pad_token_id] * (window_size - actual_len)
                window_tokens = window_tokens + padding

            input_ids = torch.tensor([window_tokens], device=device)
            attention_mask = torch.ones_like(input_ids)
            attention_mask[0, actual_len:] = 0

            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits[0], dim=-1)[:, 1]  # boundary class

            boundary_scores[start_idx : start_idx + actual_len] += (
                probs[:actual_len].cpu().numpy()
            )
            boundary_counts[start_idx : start_idx + actual_len] += 1

        avg_scores = boundary_scores / np.maximum(boundary_counts, 1)
        n_boundaries = int((avg_scores > 0.5).sum())
        segment_counts.append(n_boundaries + 1)

    return float(np.mean(segment_counts)) if segment_counts else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate segmenter on out-of-distribution traces"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint directory (contains best_model.pt)",
    )
    parser.add_argument(
        "--ood_data",
        required=True,
        help="Comma-separated paths to OOD trace JSONL files",
    )
    parser.add_argument(
        "--in_domain_data",
        default=None,
        help="Optional path to in-domain test data for comparison",
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
    results: List[Dict] = []

    # ------------------------------------------------------------------
    # In-domain evaluation (optional)
    # ------------------------------------------------------------------
    if args.in_domain_data:
        print(f"\n--- In-domain evaluation: {args.in_domain_data} ---")
        in_dataset = SegmentationDataset(
            args.in_domain_data, tokenizer, window_size=args.window_size
        )
        in_loader = DataLoader(
            in_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        metrics = evaluate_boundary_f1(model, in_loader, device)
        avg_segments = count_segments_per_trace(
            model, in_dataset, device, args.window_size
        )
        dataset_name = Path(args.in_domain_data).stem
        print(
            f"  Boundary F1: {metrics['f1']:.4f}  "
            f"(P={metrics['precision']:.4f}, R={metrics['recall']:.4f})"
        )
        print(f"  Avg segments/trace: {avg_segments:.1f}")
        print(f"  N traces: {len(in_dataset)}")
        results.append(
            {
                "dataset": dataset_name,
                "boundary_f1": round(metrics["f1"], 4),
                "boundary_precision": round(metrics["precision"], 4),
                "boundary_recall": round(metrics["recall"], 4),
                "segments_per_trace": round(avg_segments, 1),
                "n_traces": len(in_dataset),
                "domain": "in_domain",
            }
        )

    # ------------------------------------------------------------------
    # OOD evaluation
    # ------------------------------------------------------------------
    ood_paths = [p.strip() for p in args.ood_data.split(",") if p.strip()]
    for ood_path in ood_paths:
        print(f"\n--- OOD evaluation: {ood_path} ---")
        if not os.path.isfile(ood_path):
            print(f"  WARNING: file not found, skipping: {ood_path}")
            continue

        ood_dataset = SegmentationDataset(
            ood_path, tokenizer, window_size=args.window_size
        )
        ood_loader = DataLoader(
            ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        metrics = evaluate_boundary_f1(model, ood_loader, device)
        avg_segments = count_segments_per_trace(
            model, ood_dataset, device, args.window_size
        )
        dataset_name = Path(ood_path).stem
        print(
            f"  Boundary F1: {metrics['f1']:.4f}  "
            f"(P={metrics['precision']:.4f}, R={metrics['recall']:.4f})"
        )
        print(f"  Avg segments/trace: {avg_segments:.1f}")
        print(f"  N traces: {len(ood_dataset)}")
        results.append(
            {
                "dataset": dataset_name,
                "boundary_f1": round(metrics["f1"], 4),
                "boundary_precision": round(metrics["precision"], 4),
                "boundary_recall": round(metrics["recall"], 4),
                "segments_per_trace": round(avg_segments, 1),
                "n_traces": len(ood_dataset),
                "domain": "ood",
            }
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_dir = os.path.dirname(args.output)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'Dataset':<20} {'Domain':<12} {'F1':>8} {'Seg/Trace':>12} {'N':>6}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['dataset']:<20} {r['domain']:<12} "
            f"{r['boundary_f1']:>8.4f} {r['segments_per_trace']:>12.1f} "
            f"{r['n_traces']:>6}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()

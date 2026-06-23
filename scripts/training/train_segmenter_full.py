#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Full Training Script for Reasoning Step Segmentation Model

Standalone argparse-based training script (no Hydra dependency) for
StepBoundaryClassifier.  Supports standard training, hidden-dim sweeps,
and feature-ablation experiments.

Usage:
    # Standard training
    python scripts/training/train_segmenter_full.py \
        --data data/segmentation/train.jsonl \
        --test_data data/segmentation/test.jsonl \
        --output checkpoints/segmentation \
        --log_file logs/segmentation.jsonl

    # Hidden-dim sweep
    python scripts/training/train_segmenter_full.py \
        --data data/segmentation/train.jsonl \
        --test_data data/segmentation/test.jsonl \
        --output checkpoints/segmentation_sweep \
        --sweep_hidden_dim 128,256,512

    # Feature ablation
    python scripts/training/train_segmenter_full.py \
        --data data/segmentation/train.jsonl \
        --test_data data/segmentation/test.jsonl \
        --output checkpoints/segmentation_ablation \
        --ablate_feature discourse_markers
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from realm_retrieve.models.segmentation import StepBoundaryClassifier


# ---------------------------------------------------------------------------
# Feature ablation masks
# ---------------------------------------------------------------------------

DISCOURSE_MARKERS = [
    "however", "moreover", "furthermore", "therefore", "thus",
    "hence", "consequently", "meanwhile", "nevertheless", "nonetheless",
]

LOGICAL_CONNECTIVES = [
    "and", "but", "or", "if", "then",
    "because", "since", "although", "while", "unless",
]

TOPIC_SHIFT_PHRASES = [
    "on the other hand", "in contrast", "moving on",
    "turning to", "as for",
]


def ablate_text(text: str, feature_name: str) -> str:
    """Remove or mask the specified feature type from *text*.

    Args:
        text: Raw input text.
        feature_name: One of ``discourse_markers``, ``logical_connectives``,
            ``topic_shift``, ``punctuation``.

    Returns:
        Cleaned text with the target feature masked out.
    """
    if feature_name == "discourse_markers":
        pattern = r"\b(" + "|".join(re.escape(w) for w in DISCOURSE_MARKERS) + r")\b"
        return re.sub(pattern, "[DISCOURSE]", text, flags=re.IGNORECASE)

    if feature_name == "logical_connectives":
        pattern = r"\b(" + "|".join(re.escape(w) for w in LOGICAL_CONNECTIVES) + r")\b"
        return re.sub(pattern, "[LOGIC]", text, flags=re.IGNORECASE)

    if feature_name == "topic_shift":
        pattern = "|".join(re.escape(p) for p in TOPIC_SHIFT_PHRASES)
        return re.sub(pattern, "[TOPIC]", text, flags=re.IGNORECASE)

    if feature_name == "punctuation":
        return re.sub(r"[^\w\s]", "[PUNCT]", text)

    raise ValueError(
        f"Unknown ablation feature: {feature_name!r}. "
        "Choose from: discourse_markers, logical_connectives, topic_shift, punctuation"
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """Dataset for reasoning step boundary detection.

    Loads JSONL data where each line contains ``text`` and
    ``boundary_labels`` fields.  Optionally applies feature ablation
    before tokenization.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        window_size: int = 128,
        ablate_feature: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.ablate_feature = ablate_feature

        self.samples = self._load_data(data_path)

    # ------------------------------------------------------------------

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load human-annotated reasoning traces from JSONL."""
        samples: List[Dict] = []
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

        text = sample["text"]
        if self.ablate_feature is not None:
            text = ablate_text(text, self.ablate_feature)

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        labels = sample["boundary_labels"]  # [0, 1, 0, 0, 1, ...]

        # Pad / truncate to window_size
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
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch and return loss / F1 metrics."""
    model.train()
    total_loss = 0.0
    tp, fp, fn = 0, 0, 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)  # [batch, seq_len, 2]

        logits_flat = logits.view(-1, 2)
        labels_flat = labels.view(-1)

        loss = criterion(logits_flat, labels_flat)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits_flat, dim=-1)
        tp += ((preds == 1) & (labels_flat == 1)).sum().item()
        fp += ((preds == 1) & (labels_flat == 0)).sum().item()
        fn += ((preds == 0) & (labels_flat == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataset and return loss / F1 metrics."""
    model.eval()
    total_loss = 0.0
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
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
        "loss": total_loss / max(len(dataloader), 1),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def run_training(
    args: argparse.Namespace,
    hidden_dim: int,
    ablate_feature: Optional[str] = None,
    tag: str = "",
) -> Dict[str, float]:
    """Train a single StepBoundaryClassifier and return final test metrics.

    This is the shared core used by standard, sweep, and ablation modes.
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Model --------------------------------------------------------
    model = StepBoundaryClassifier(
        base_model=args.base_model,
        hidden_dim=hidden_dim,
        num_layers=args.n_layers,
        num_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ---- Data ---------------------------------------------------------
    train_dataset = SegmentationDataset(
        args.data,
        model.tokenizer,
        window_size=args.window_size,
        ablate_feature=ablate_feature,
    )
    test_dataset = SegmentationDataset(
        args.test_data,
        model.tokenizer,
        window_size=args.window_size,
        ablate_feature=ablate_feature,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ---- Optimizer / loss ---------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ---- Output dirs --------------------------------------------------
    output_dir = Path(args.output)
    if tag:
        output_dir = output_dir / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop ------------------------------------------------
    best_f1 = 0.0
    best_test_metrics: Dict[str, float] = {}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        print(
            f"  Train Loss: {train_metrics['loss']:.4f}  "
            f"F1: {train_metrics['f1']:.4f}"
        )

        test_metrics = evaluate(model, test_loader, criterion, device)
        print(
            f"  Test  Loss: {test_metrics['loss']:.4f}  "
            f"F1: {test_metrics['f1']:.4f}  "
            f"P: {test_metrics['precision']:.4f}  "
            f"R: {test_metrics['recall']:.4f}"
        )

        # Per-epoch checkpoint
        epoch_dir = output_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1": test_metrics["f1"],
            },
            epoch_dir / "model.pt",
        )

        # Best model checkpoint
        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            best_test_metrics = dict(test_metrics)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "f1": best_f1,
                },
                output_dir / "best_model.pt",
            )
            print(f"  -> Saved new best model (F1: {best_f1:.4f})")

        # JSONL log
        if args.log_file:
            log_entry = {
                "epoch": epoch,
                "step": epoch * len(train_loader),
                "train_loss": train_metrics["loss"],
                "train_f1": train_metrics["f1"],
                "test_f1": test_metrics["f1"],
                "lr": args.lr,
                "metric_name": "f1",
                "metric_value": test_metrics["f1"],
            }
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

    print(f"\nTraining complete! Best test F1: {best_f1:.4f}")
    return {
        "best_f1": best_f1,
        "trainable_params": trainable_params,
        **best_test_metrics,
    }


# ---------------------------------------------------------------------------
# Mode entry-points
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace) -> None:
    """Train separate models for each hidden_dim value and report results."""
    dims = [int(d.strip()) for d in args.sweep_hidden_dim.split(",")]
    print(f"Running hidden-dim sweep: {dims}")

    results: List[Dict] = []
    for dim in dims:
        print(f"\n{'='*60}")
        print(f"Sweep: hidden_dim={dim}")
        print(f"{'='*60}")
        metrics = run_training(args, hidden_dim=dim, tag=f"sweep_dim_{dim}")
        entry = {"hidden_dim": dim, **metrics}
        results.append(entry)
        print(f"  hidden_dim={dim} -> test_f1={metrics['best_f1']:.4f}, "
              f"params={metrics['trainable_params']:,}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep results saved to {results_path}")


def run_ablation(args: argparse.Namespace) -> None:
    """Train with one feature type ablated and save results."""
    feature = args.ablate_feature
    print(f"\n{'='*60}")
    print(f"Ablation: masking feature={feature!r}")
    print(f"{'='*60}")

    metrics = run_training(
        args,
        hidden_dim=args.hidden_dim,
        ablate_feature=feature,
        tag=f"ablation_{feature}",
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"ablation_{feature}.json"
    result = {"ablated_feature": feature, **metrics}
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nAblation results saved to {results_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train StepBoundaryClassifier (standalone, no Hydra)",
    )

    # Data
    parser.add_argument("--data", required=True, help="Training data JSONL path")
    parser.add_argument("--test_data", required=True, help="Test data JSONL path")

    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension (default: 256)")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of transformer layers (default: 3)")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads (default: 4)")
    parser.add_argument("--base_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Base model name (default: sentence-transformers/all-MiniLM-L6-v2)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--window_size", type=int, default=128, help="Sliding window size (default: 128)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Output
    parser.add_argument("--output", required=True, help="Checkpoint output directory")
    parser.add_argument("--log_file", type=str, default=None, help="JSONL log file path for per-epoch metrics")

    # Sweep mode
    parser.add_argument("--sweep_hidden_dim", type=str, default=None,
                        help='Comma-separated dims for hidden dim sweep (e.g., "128,256,512")')

    # Ablation mode
    parser.add_argument("--ablate_feature", type=str, default=None,
                        choices=["discourse_markers", "logical_connectives", "topic_shift", "punctuation"],
                        help="Feature name to ablate")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sweep_hidden_dim:
        run_sweep(args)
    elif args.ablate_feature:
        run_ablation(args)
    else:
        run_training(args, hidden_dim=args.hidden_dim)


if __name__ == "__main__":
    main()

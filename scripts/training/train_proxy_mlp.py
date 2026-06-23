#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Training script for the Verbalized Uncertainty Proxy MLP.

Loads annotated reasoning traces, extracts SBERT embeddings and surface
features, and trains the 2-layer MLP with early stopping on dev AUROC.

Usage:
    python scripts/training/train_proxy_mlp.py \
        --traces data/annotated_traces.jsonl \
        --output checkpoints/proxy_mlp \
        --epochs 50 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from realm_retrieve.models.proxy_mlp import ProxyMLP


def load_traces(path: str) -> List[Dict]:
    """Load annotated traces (one JSON object per line)."""
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def extract_features(
    traces: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract SBERT embeddings, surface features, running U_ent, and labels.

    Each trace is expected to have:
        - "steps": list of {"text": str, "u_ent": float, "label": int}
        - OR flat fields "text", "u_ent", "label" for single-step traces
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    all_texts = []
    all_surface = []
    all_u_ent_running = []
    all_labels = []

    for trace in traces:
        steps = trace.get("steps", [trace])
        recent_u_ent: List[float] = []

        for step in steps:
            text = step["text"]
            label = step.get("label", 0)

            all_texts.append(text)
            seg_len, hedge_count, ent_count = ProxyMLP.extract_surface_features(text)
            all_surface.append([float(seg_len), float(hedge_count), float(ent_count)])

            u_ent = step.get("u_ent", 0.0)
            recent_u_ent.append(u_ent)
            window = recent_u_ent[-3:]
            all_u_ent_running.append([float(np.mean(window))])
            all_labels.append(label)

    embeddings = encoder.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)

    return (
        np.array(embeddings, dtype=np.float32),
        np.array(all_surface, dtype=np.float32),
        np.array(all_u_ent_running, dtype=np.float32),
        np.array(all_labels, dtype=np.float32),
    )


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC from binary labels and predicted scores."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def train(
    train_emb: np.ndarray,
    train_surf: np.ndarray,
    train_uent: np.ndarray,
    train_labels: np.ndarray,
    dev_emb: np.ndarray,
    dev_surf: np.ndarray,
    dev_uent: np.ndarray,
    dev_labels: np.ndarray,
    output_dir: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 5,
) -> Dict[str, List[float]]:
    """Train ProxyMLP with early stopping on dev AUROC."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ProxyMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_emb),
        torch.from_numpy(train_surf),
        torch.from_numpy(train_uent),
        torch.from_numpy(train_labels).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    dev_emb_t = torch.from_numpy(dev_emb).to(device)
    dev_surf_t = torch.from_numpy(dev_surf).to(device)
    dev_uent_t = torch.from_numpy(dev_uent).to(device)

    history: Dict[str, List[float]] = {"train_loss": [], "dev_auroc": []}
    best_auroc = -1.0
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for emb, surf, uent, lbl in train_loader:
            emb, surf, uent, lbl = (
                emb.to(device), surf.to(device), uent.to(device), lbl.to(device),
            )
            pred = model(emb, surf, uent)
            loss = criterion(pred, lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        model.eval()
        with torch.no_grad():
            dev_pred = model(dev_emb_t, dev_surf_t, dev_uent_t).cpu().numpy().squeeze()
        auroc = compute_auroc(dev_labels, dev_pred)
        history["dev_auroc"].append(auroc)

        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | dev_auroc={auroc:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            wait = 0
            model.save_pretrained(output_dir)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (best dev AUROC={best_auroc:.4f})")
                break

    with open(os.path.join(output_dir, "training_curves.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


def train_with_logging(
    train_emb: np.ndarray,
    train_surf: np.ndarray,
    train_uent: np.ndarray,
    train_labels: np.ndarray,
    dev_emb: np.ndarray,
    dev_surf: np.ndarray,
    dev_uent: np.ndarray,
    dev_labels: np.ndarray,
    test_emb: np.ndarray | None,
    test_surf: np.ndarray | None,
    test_uent: np.ndarray | None,
    test_labels: np.ndarray | None,
    output_dir: str,
    log_file: str | None = None,
    hidden_dim: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 5,
) -> Dict[str, List[float]]:
    """Train ProxyMLP with JSONL logging and separate test set evaluation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ProxyMLP(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_emb),
        torch.from_numpy(train_surf),
        torch.from_numpy(train_uent),
        torch.from_numpy(train_labels).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    dev_emb_t = torch.from_numpy(dev_emb).to(device)
    dev_surf_t = torch.from_numpy(dev_surf).to(device)
    dev_uent_t = torch.from_numpy(dev_uent).to(device)

    has_test = test_emb is not None
    if has_test:
        test_emb_t = torch.from_numpy(test_emb).to(device)
        test_surf_t = torch.from_numpy(test_surf).to(device)
        test_uent_t = torch.from_numpy(test_uent).to(device)

    history: Dict[str, List[float]] = {"train_loss": [], "dev_auroc": [], "test_auroc": []}
    best_auroc = -1.0
    best_epoch = 0
    wait = 0

    log_fh = None
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_file, "w")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for emb_b, surf_b, uent_b, lbl_b in train_loader:
            emb_b, surf_b, uent_b, lbl_b = (
                emb_b.to(device), surf_b.to(device), uent_b.to(device), lbl_b.to(device),
            )
            pred = model(emb_b, surf_b, uent_b)
            loss = criterion(pred, lbl_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        model.eval()
        with torch.no_grad():
            dev_pred = model(dev_emb_t, dev_surf_t, dev_uent_t).cpu().numpy().squeeze()
        dev_auc = compute_auroc(dev_labels, dev_pred)
        history["dev_auroc"].append(dev_auc)

        # Compute train AUROC
        with torch.no_grad():
            train_emb_t = torch.from_numpy(train_emb).to(device)
            train_surf_t = torch.from_numpy(train_surf).to(device)
            train_uent_t = torch.from_numpy(train_uent).to(device)
            train_pred = model(train_emb_t, train_surf_t, train_uent_t).cpu().numpy().squeeze()
        train_auc = compute_auroc(train_labels, train_pred)

        test_auc = 0.0
        if has_test:
            with torch.no_grad():
                test_pred = model(test_emb_t, test_surf_t, test_uent_t).cpu().numpy().squeeze()
            test_auc = compute_auroc(test_labels, test_pred)
        history["test_auroc"].append(test_auc)

        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | dev_auroc={dev_auc:.4f} | test_auroc={test_auc:.4f}")

        if log_fh:
            log_entry = {
                "epoch": epoch,
                "step": epoch * max(n_batches, 1),
                "train_loss": round(avg_loss, 4),
                "train_auroc": round(train_auc, 4),
                "dev_auroc": round(dev_auc, 4),
                "test_auroc": round(test_auc, 4),
                "lr": lr,
                "metric_name": "dev_auroc",
                "metric_value": round(dev_auc, 4),
            }
            log_fh.write(json.dumps(log_entry) + "\n")
            log_fh.flush()

        if dev_auc > best_auroc:
            best_auroc = dev_auc
            best_epoch = epoch
            wait = 0
            model.save_pretrained(output_dir)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (best dev AUROC={best_auroc:.4f} at epoch {best_epoch})")
                break

    if log_fh:
        log_fh.close()

    with open(os.path.join(output_dir, "training_curves.json"), "w") as f:
        json.dump(history, f, indent=2)

    final_info = {
        "train_auroc": round(train_auc, 2),
        "dev_auroc": round(best_auroc, 2),
        "test_auroc": round(test_auc, 2),
        "best_epoch": best_epoch,
    }
    with open(os.path.join(output_dir, "proxy_mlp_final.json"), "w") as f:
        json.dump(final_info, f, indent=2)

    return history


def main():
    parser = argparse.ArgumentParser(
        description="Train Proxy MLP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--traces", default=None, help="Path to annotated traces JSONL (train+dev combined)")
    parser.add_argument("--data", default=None, help="Path to training traces JSONL")
    parser.add_argument("--test_data", default=None, help="Path to separate test traces JSONL")
    parser.add_argument("--output", default="checkpoints/proxy_mlp", help="Output directory")
    parser.add_argument("--hidden_dim", type=int, default=128, help="MLP hidden dimension")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dev-split", type=float, default=0.15)
    parser.add_argument("--log_file", default=None, help="Path to JSONL log file for per-epoch metrics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    data_path = args.data or args.traces
    if data_path is None:
        parser.error("Either --data or --traces must be specified")

    print("Loading traces ...")
    traces = load_traces(data_path)
    print(f"  {len(traces)} traces loaded")

    print("Extracting features ...")
    emb, surf, uent, labels = extract_features(traces)
    print(f"  {len(labels)} samples, {int(labels.sum())} positive")

    test_emb = test_surf = test_uent = test_labels = None
    if args.test_data:
        print("Loading test traces ...")
        test_traces = load_traces(args.test_data)
        print(f"  {len(test_traces)} test traces loaded")
        print("Extracting test features ...")
        test_emb, test_surf, test_uent, test_labels = extract_features(test_traces)
        print(f"  {len(test_labels)} test samples, {int(test_labels.sum())} positive")

    n = len(labels)
    split = int(n * (1 - args.dev_split))
    indices = np.random.default_rng(args.seed).permutation(n)
    train_idx, dev_idx = indices[:split], indices[split:]

    print(f"Training ({len(train_idx)} train / {len(dev_idx)} dev) ...")
    train_with_logging(
        emb[train_idx], surf[train_idx], uent[train_idx], labels[train_idx],
        emb[dev_idx], surf[dev_idx], uent[dev_idx], labels[dev_idx],
        test_emb=test_emb,
        test_surf=test_surf,
        test_uent=test_uent,
        test_labels=test_labels,
        output_dir=args.output,
        log_file=args.log_file,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    print(f"Done — checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()

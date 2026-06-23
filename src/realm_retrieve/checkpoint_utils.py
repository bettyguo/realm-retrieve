"""Utilities for loading both standard and protected (sharded) checkpoints."""

import json
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def load_checkpoint(path: str, map_location: str = "cpu"):
    """Load checkpoint from a .pt file or a protected sharded directory.

    Protected format: directory with manifest.json + model_shard_XXXX.pt files.
    Standard format: single .pt file or directory containing model.pt.
    """
    if os.path.isdir(path):
        manifest_path = os.path.join(path, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            fmt = manifest["checkpoint_format"]
            shard_idx = fmt["weight_map"]["core_weights_shard"]
            core_key = fmt["weight_map"]["core_weights_key"]
            shard_path = os.path.join(
                path, f"model_shard_{shard_idx:04d}.pt"
            )
            shard = torch.load(
                shard_path, map_location=map_location, weights_only=True
            )
            return shard[core_key]
        model_pt = os.path.join(path, "model.pt")
        if os.path.exists(model_pt):
            return torch.load(
                model_pt, map_location=map_location, weights_only=True
            )
        raise FileNotFoundError(f"No checkpoint found in {path}")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except RuntimeError:
        logger.warning(
            "weights_only=True failed for %s, retrying with weights_only=False",
            path,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def validate_checkpoint_embedding_dim(
    state_dict: dict,
    expected_dim: int,
    key: str = "query_projector.weight",
) -> None:
    """Validate that a policy checkpoint's embedding dimension matches expectations.

    Raises RuntimeError with a clear message when the checkpoint was trained
    with a different embedding dimension than the model being loaded into.
    """
    weights = state_dict.get(key)
    if weights is None:
        sd = state_dict.get("model_state_dict", state_dict)
        weights = sd.get(key)
    if weights is None:
        return
    actual_dim = weights.shape[1]
    if actual_dim != expected_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: checkpoint has {key} with "
            f"input_features={actual_dim}, but the model expects "
            f"embedding_dim={expected_dim}. Ensure the embedding encoder "
            f"and policy model use the same dimension."
        )

# SPDX-License-Identifier: Apache-2.0
"""
Query Generation Module

Single-layer transformer decoder that formulates dense retrieval queries
from the current reasoning step and question prefix embeddings.

Architecture: TransformerDecoderLayer(d_model=512, nhead=8) -> mean-pool -> Linear(512, output_dim) -> L2-norm
"""

from __future__ import annotations

import json
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryGen(nn.Module):
    """
    Learned query generator for dense retrieval during reasoning.

    Replaces the naive ``f"{question} {step.text[:200]}"`` query used in
    the baseline ``evaluate.py`` with a cross-attention query formulator
    that outputs a 768-d L2-normalized vector matching the ColBERTv2
    query embedding space.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        n_heads: int = 8,
        output_dim: int = 768,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.output_dim = output_dim

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )

        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        reasoning_step_emb: torch.Tensor,
        query_prefix_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            reasoning_step_emb: [B, L_step, hidden_dim] from policy encoder
            query_prefix_emb: [B, L_prefix, hidden_dim] from question encoder

        Returns:
            query_vector: [B, output_dim] L2-normalized dense retrieval query
        """
        decoded = self.decoder_layer(
            tgt=query_prefix_emb,
            memory=reasoning_step_emb,
        )

        pooled = decoded.mean(dim=1)

        projected = self.output_projection(pooled)

        query_vector = F.normalize(projected, p=2, dim=-1)
        return query_vector

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        config = {
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "output_dim": self.output_dim,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "QueryGen":
        from realm_retrieve.checkpoint_utils import load_checkpoint

        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = load_checkpoint(path, map_location=device)
        model.load_state_dict(state_dict)
        return model

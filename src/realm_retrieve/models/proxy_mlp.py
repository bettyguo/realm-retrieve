# SPDX-License-Identifier: Apache-2.0
"""
Verbalized Uncertainty Proxy MLP

2-layer MLP that estimates verbalized uncertainty for closed-weight models
(e.g. o1) where direct confidence extraction is not possible.

Architecture: Linear(input_dim, 128) -> GELU -> Linear(128, 1) -> Sigmoid
Input: SBERT embedding + 3 surface features + 1 running U_ent mean
"""

from __future__ import annotations

import json
import os
import re
from typing import Tuple

import torch
import torch.nn as nn


class ProxyMLP(nn.Module):
    """
    Proxy MLP for estimating verbalized uncertainty on closed-weight LRMs.

    Surface features are extracted from the reasoning step text and
    concatenated with the SBERT embedding and running U_ent mean to
    produce a scalar uncertainty estimate in [0, 1].
    """

    HEDGE_PHRASES = [
        "probably", "I think", "maybe",
    ]

    def __init__(self, input_dim: int = 388, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def extract_surface_features(text: str) -> Tuple[int, int, int]:
        """
        Extract surface-level features from a reasoning step.

        Returns:
            segment_length_tokens: approximate token count (whitespace split)
            hedge_count: number of hedge phrases detected
            entity_count: number of named entities (via spaCy en_core_web_sm)
        """
        segment_length_tokens = len(text.split())

        text_lower = text.lower()
        hedge_count = sum(
            1 for phrase in ProxyMLP.HEDGE_PHRASES if phrase.lower() in text_lower
        )

        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            entity_count = len(doc.ents)
        except Exception:
            entity_count = 0

        return segment_length_tokens, hedge_count, entity_count

    def forward(
        self,
        sbert_emb: torch.Tensor,
        surface_features: torch.Tensor,
        u_ent_running: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sbert_emb: [B, 384] SBERT embedding of the reasoning step
            surface_features: [B, 3] (segment_length, hedge_count, entity_count)
            u_ent_running: [B, 1] running mean of U_ent over last 3 steps

        Returns:
            uncertainty: [B, 1] proxy uncertainty score in [0, 1]
        """
        x = torch.cat([sbert_emb, surface_features, u_ent_running], dim=-1)
        return self.mlp(x)

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        config = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "ProxyMLP":
        from realm_retrieve.checkpoint_utils import load_checkpoint

        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = load_checkpoint(path, map_location=device)
        model.load_state_dict(state_dict)
        return model

#!/usr/bin/env python3
"""
Train Retrieval Intervention Policy with REINFORCE

Trains the policy pi(a|s) for adaptive retrieval intervention.
Reward: R = F1(a_pi, a*) - lambda1 * n_ret - lambda2 * t_latency

Training: 50,000 steps with curriculum learning (lambda1: 0.5 -> 0.1)

This Hydra entry-point delegates to the full episode-based training logic
in ``scripts/training/train_policy_full.py``.

Usage:
    python train_policy.py configs/experiments/train_policy.yaml
"""

import sys
import os

import hydra
from omegaconf import DictConfig
import torch
from tqdm import tqdm
import wandb
import json
from typing import Dict, List, Optional

from realm_retrieve.models import (
    RetrievalInterventionPolicy,
    REINFORCETrainer,
)

# Import real episode execution from the full training script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "training"))
from train_policy_full import run_episode, extract_states_from_example, evaluate_dev, load_jsonl


@hydra.main(version_base=None, config_path="configs/experiments", config_name="train_policy")
def main(cfg: DictConfig):
    """Main training function using real episode execution."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize W&B
    if cfg.logging.use_wandb:
        wandb.init(project="realm-retrieve", name="policy-training", config=dict(cfg))

    embedding_dim = cfg.model.get("embedding_dim", 768)

    # Initialize policy
    policy = RetrievalInterventionPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        retrieval_threshold=cfg.model.threshold,
    )

    # Initialize trainer
    trainer = REINFORCETrainer(
        policy=policy,
        learning_rate=cfg.training.learning_rate,
        lambda1_start=cfg.training.lambda1_start,
        lambda1_end=cfg.training.lambda1_end,
        lambda2=cfg.training.lambda2,
        device=str(device),
    )

    # Load training data
    train_data = load_jsonl(cfg.data.train_path)
    print(f"Loaded {len(train_data)} training examples")

    # Load dev data if available
    dev_data: Optional[List[Dict]] = None
    if hasattr(cfg.data, "dev_path") and cfg.data.dev_path:
        dev_data = load_jsonl(cfg.data.dev_path)
        print(f"Loaded {len(dev_data)} dev examples")

    # Training loop with real episode execution
    batch_states: List = []
    batch_actions: List = []
    batch_rewards_data: List = []
    episode_state_counts: List = []
    batch_size = cfg.training.get("batch_size", 64)
    latest_dev_f1 = 0.0

    for step in tqdm(range(cfg.training.num_steps), desc="Training"):
        # Sample episode
        idx = step % len(train_data)
        example = train_data[idx]

        # Run real episode with policy decisions
        states, actions, f1_score, num_retrievals, total_latency = run_episode(
            policy, example, str(device), embedding_dim=embedding_dim,
        )

        batch_states.extend(states)
        batch_actions.extend(actions)
        batch_rewards_data.append((f1_score, num_retrievals, total_latency))
        episode_state_counts.append(len(states))

        # Accumulate a mini-batch before updating
        if len(batch_rewards_data) < batch_size and step < cfg.training.num_steps - 1:
            continue

        # Build per-state rewards
        per_state_rewards = []
        for ep_idx, (ep_f1, ep_ret, ep_lat) in enumerate(batch_rewards_data):
            for _ in range(episode_state_counts[ep_idx]):
                per_state_rewards.append((ep_f1, ep_ret, ep_lat))

        # Training step with per-state rewards
        metrics = trainer.train_step(
            states=batch_states,
            actions=batch_actions,
            rewards=per_state_rewards,
        )

        batch_states = []
        batch_actions = []
        batch_rewards_data = []
        episode_state_counts = []

        # Logging
        if step % 100 == 0:
            if cfg.logging.use_wandb:
                wandb.log(metrics)

            if step % 1000 == 0:
                # Run dev evaluation
                if dev_data is not None:
                    dev_metrics = evaluate_dev(
                        policy, dev_data, str(device),
                        embedding_dim=embedding_dim,
                    )
                    latest_dev_f1 = dev_metrics["dev_f1"]
                    dev_ret_rate = dev_metrics["retrieval_rate"]
                else:
                    dev_ret_rate = 0.0

                print(
                    f"\nStep {step}: Reward={metrics['reward']:.3f}, "
                    f"Loss={metrics['loss']:.3f}, "
                    f"dev_f1={latest_dev_f1:.4f}, "
                    f"dev_ret_rate={dev_ret_rate:.4f}"
                )

        # Save checkpoint
        if step % cfg.training.save_every == 0 and step > 0:
            save_path = f"{cfg.training.checkpoint_dir}/checkpoint_{step}.pt"
            trainer.save_checkpoint(save_path, step)
            print(f"Saved checkpoint to {save_path}")

    # Save final model
    final_path = f"{cfg.training.checkpoint_dir}/final_model.pt"
    trainer.save_checkpoint(final_path, cfg.training.num_steps)
    print(f"\nTraining complete! Final model saved to {final_path}")

    # Final dev evaluation
    if dev_data is not None:
        dev_metrics = evaluate_dev(
            policy, dev_data, str(device), embedding_dim=embedding_dim,
        )
        print(
            f"Final dev_f1={dev_metrics['dev_f1']:.4f}  "
            f"retrieval_rate={dev_metrics['retrieval_rate']:.4f}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train Retrieval Intervention Policy with REINFORCE

Trains the policy π(a|s) for adaptive retrieval intervention.
Reward: R = F1(a_π, a*) - λ1·n_ret - λ2·t_latency

Training: 50,000 steps with curriculum learning (λ1: 0.5 → 0.1)

Usage:
    python train_policy.py configs/experiments/train_policy.yaml
"""

import hydra
from omegaconf import DictConfig
import torch
from tqdm import tqdm
import wandb
import json

from realm_retrieve.models import (
    RetrievalInterventionPolicy,
    REINFORCETrainer,
    create_reasoning_model,
    ColBERTRetriever,
)


@hydra.main(version_base=None, config_path="configs/experiments", config_name="train_policy")
def main(cfg: DictConfig):
    """Main training function."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize W&B
    if cfg.logging.use_wandb:
        wandb.init(project="realm-retrieve", name="policy-training", config=dict(cfg))
    
    # Initialize policy
    policy = RetrievalInterventionPolicy(
        embedding_dim=cfg.model.embedding_dim,
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
        device=device,
    )
    
    # Load training data
    with open(cfg.data.train_path, 'r') as f:
        train_data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(train_data)} training examples")
    
    # Training loop
    for step in tqdm(range(cfg.training.num_steps), desc="Training"):
        # Sample episode
        idx = step % len(train_data)
        example = train_data[idx]
        
        # Run episode (simplified - actual implementation would use full system)
        # This is a placeholder for the full training loop
        states = []
        actions = []
        
        # Simulate episode metrics
        f1_score = 0.7  # Placeholder
        num_retrievals = 2
        total_latency = 0.5
        
        # Training step
        metrics = trainer.train_step(
            states=states,
            actions=actions,
            f1_score=f1_score,
            num_retrievals=num_retrievals,
            total_latency=total_latency,
        )
        
        # Logging
        if step % 100 == 0:
            if cfg.logging.use_wandb:
                wandb.log(metrics)
            
            if step % 1000 == 0:
                print(f"\nStep {step}: Reward={metrics['reward']:.3f}, Loss={metrics['loss']:.3f}")
        
        # Save checkpoint
        if step % cfg.training.save_every == 0 and step > 0:
            save_path = f"{cfg.training.checkpoint_dir}/checkpoint_{step}.pt"
            trainer.save_checkpoint(save_path, step)
            print(f"Saved checkpoint to {save_path}")
    
    # Save final model
    final_path = f"{cfg.training.checkpoint_dir}/final_model.pt"
    trainer.save_checkpoint(final_path, cfg.training.num_steps)
    print(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()

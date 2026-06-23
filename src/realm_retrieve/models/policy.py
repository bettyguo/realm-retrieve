"""
Retrieval Intervention Policy

REINFORCE-trained policy for deciding when to retrieve during reasoning.

Policy π(a_t | s_t) outputs:
1. Retrieve decision: a_ret ∈ {0, 1}
2. Query formulation: q_i if retrieve
3. Context integration: how to merge retrieved docs

Reward: R = F1(a_π, a*) - λ1·n_ret - λ2·t_latency

REINFORCE with curriculum learning (λ1: 0.5 → 0.1).
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np


@dataclass
class PolicyState:
    """State representation for retrieval policy."""
    query_embedding: torch.Tensor  # Original query embedding
    current_step_embedding: torch.Tensor  # Current reasoning step
    rsus_features: torch.Tensor  # RSUS components (α, β, γ)
    retrieval_history: torch.Tensor  # History features
    step_number: int
    total_steps_estimate: int


@dataclass
class PolicyAction:
    """Action output from policy."""
    should_retrieve: bool
    retrieve_probability: float
    query_embedding: Optional[torch.Tensor] = None
    prob_tensor: Optional[torch.Tensor] = None  # Graph-connected probability for gradient flow


class RetrievalInterventionPolicy(nn.Module):
    """
    Learned policy for adaptive retrieval intervention.
    
    Architecture:
    - State encoder: Processes query, current step, RSUS, history
    - Transformer policy network: 4 layers, 512 hidden dim, 8 heads
    - Retrieval head: Binary decision (retrieve / continue)
    - Query generator: Generates query embedding if retrieving
    
    Training: REINFORCE with baseline
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        retrieval_threshold: float = 0.65,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.retrieval_threshold = retrieval_threshold

        # State encoder: Project different inputs to common dimension
        self.query_projector = nn.Linear(embedding_dim, hidden_dim)
        self.step_projector = nn.Linear(embedding_dim, hidden_dim)
        self.rsus_projector = nn.Linear(3, hidden_dim)  # 3 RSUS components
        self.history_projector = nn.Linear(10, hidden_dim)  # History features
        
        # Positional encoding for step number
        self.step_embedding = nn.Embedding(200, hidden_dim)  # Max 200 steps
        
        # Transformer policy network
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.policy_network = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Retrieval decision head
        self.retrieval_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output probability
        )
        
        # Query generator (cross-attention to current step)
        self.query_generator = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.query_output = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(
        self,
        state: PolicyState,
        deterministic: bool = False,
    ) -> Tuple[PolicyAction, Dict[str, torch.Tensor]]:
        """
        Forward pass through policy network.
        
        Args:
            state: Current policy state
            deterministic: If True, use threshold; else sample from distribution
            
        Returns:
            action: Policy action
            info: Additional info for training (logits, value estimates, etc.)
        """
        # Encode state components
        query_enc = self.query_projector(state.query_embedding)  # [hidden_dim]
        step_enc = self.step_projector(state.current_step_embedding)  # [hidden_dim]
        rsus_enc = self.rsus_projector(state.rsus_features)  # [hidden_dim]
        history_enc = self.history_projector(state.retrieval_history)  # [hidden_dim]
        step_pos = self.step_embedding(torch.tensor(state.step_number))  # [hidden_dim]
        
        # Combine into sequence [batch=1, seq_len=5, hidden_dim]
        state_sequence = torch.stack([
            query_enc,
            step_enc,
            rsus_enc,
            history_enc,
            step_pos,
        ]).unsqueeze(0)
        
        # Apply policy network
        policy_output = self.policy_network(state_sequence)  # [1, 5, hidden_dim]
        
        # Pool to get state representation
        state_repr = policy_output.mean(dim=1)  # [1, hidden_dim]
        
        # Retrieval decision
        retrieve_prob = self.retrieval_head(state_repr).squeeze()  # Scalar
        
        if deterministic:
            should_retrieve = retrieve_prob > self.retrieval_threshold
        else:
            # Sample from Bernoulli distribution
            should_retrieve = torch.bernoulli(retrieve_prob).bool()
        
        # Generate query if retrieving
        query_embedding = None
        if should_retrieve:
            # Use current step as query for decoder
            query_input = step_enc.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
            query_output = self.query_generator(
                query_input,
                policy_output,
            )  # [1, 1, hidden_dim]
            query_embedding = F.normalize(self.query_output(query_output.squeeze(0)), p=2, dim=-1)  # [embedding_dim], L2-normalized
        
        action = PolicyAction(
            should_retrieve=should_retrieve.item(),
            retrieve_probability=retrieve_prob.item(),
            query_embedding=query_embedding,
            prob_tensor=retrieve_prob,  # Keep graph-connected tensor for entropy gradient
        )
        
        info = {
            "retrieve_logit": torch.logit(retrieve_prob),
            "state_repr": state_repr,
        }
        
        return action, info
    
    def compute_log_prob(
        self,
        state: PolicyState,
        action: PolicyAction,
    ) -> torch.Tensor:
        """
        Compute log probability of action under current policy.
        Used for REINFORCE gradient computation.

        Args:
            state: State at which action was taken
            action: Action that was taken

        Returns:
            log_prob: log π(a|s)
        """
        # Forward pass
        _, info = self.forward(state, deterministic=False)
        retrieve_logit = info["retrieve_logit"]

        # Bernoulli log probability
        if action.should_retrieve:
            log_prob = F.logsigmoid(retrieve_logit)
        else:
            log_prob = F.logsigmoid(-retrieve_logit)

        return log_prob

    def compute_log_prob_and_entropy(
        self,
        state: PolicyState,
        action: PolicyAction,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability and entropy from a single forward pass.

        Returns graph-connected tensors for both, so the entropy bonus
        provides gradient signal to the policy network.

        Returns:
            log_prob: log π(a|s)
            entropy: H(π(·|s)) = -p log(p) - (1-p) log(1-p)
        """
        _, info = self.forward(state, deterministic=False)
        retrieve_logit = info["retrieve_logit"]

        # Bernoulli log probability
        if action.should_retrieve:
            log_prob = F.logsigmoid(retrieve_logit)
        else:
            log_prob = F.logsigmoid(-retrieve_logit)

        # Binary entropy from logit (graph-connected)
        p = torch.sigmoid(retrieve_logit)
        entropy = -(p * torch.log(p + 1e-10)) - ((1 - p) * torch.log(1 - p + 1e-10))

        return log_prob, entropy


class REINFORCETrainer:
    """
    REINFORCE trainer for retrieval policy.
    
    Reward function:
        R = F1(answer, ground_truth) - λ1 * num_retrievals - λ2 * total_latency
    
    Features:
    - Curriculum learning: Start with high λ1, decrease over training
    - Baseline subtraction for variance reduction
    - Entropy regularization for exploration
    """
    
    def __init__(
        self,
        policy: RetrievalInterventionPolicy,
        learning_rate: float = 1e-4,
        lambda1_start: float = 0.5,
        lambda1_end: float = 0.1,
        lambda2: float = 0.01,
        entropy_coef: float = 0.01,
        baseline_momentum: float = 0.95,
        device: str = "cuda",
    ):
        self.policy = policy.to(device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        
        # Curriculum learning schedule for λ1
        self.lambda1_start = lambda1_start
        self.lambda1_end = lambda1_end
        self.lambda2 = lambda2
        
        self.entropy_coef = entropy_coef
        
        # Running baseline (exponential moving average of returns)
        self.baseline = 0.0
        self.baseline_momentum = baseline_momentum
        
        self.device = device
        self.training_step = 0
    
    def train_step(
        self,
        states: List[PolicyState],
        actions: List[PolicyAction],
        rewards: Optional[List[Tuple[float, float, float]]] = None,
        f1_score: float = 0.0,
        num_retrievals: int = 0,
        total_latency: float = 0.0,
    ) -> Dict[str, float]:
        """
        Single REINFORCE training step with per-state rewards.

        Args:
            states: List of states visited during episode(s)
            actions: List of actions taken
            rewards: Per-state reward tuples (f1, num_retrievals, latency).
                     When provided, each state gets its own episode's reward
                     instead of a single averaged reward. This preserves the
                     per-episode REINFORCE signal in batched training.
            f1_score: (Legacy) single F1 score -- used only when rewards is None
            num_retrievals: (Legacy) single retrieval count
            total_latency: (Legacy) single latency value

        Returns:
            metrics: Training metrics (loss, reward, etc.)
        """
        # Compute lambda1 using curriculum schedule
        progress = min(1.0, self.training_step / 50000)
        lambda1 = self.lambda1_start + (self.lambda1_end - self.lambda1_start) * progress

        # Build per-state reward and advantage arrays
        if rewards is not None:
            # Per-state rewards: each state has its own episode's reward tuple
            per_state_rewards = []
            for (f1_i, ret_i, lat_i) in rewards:
                r = f1_i - lambda1 * ret_i - self.lambda2 * lat_i
                per_state_rewards.append(r)
            mean_reward = float(np.mean(per_state_rewards)) if per_state_rewards else 0.0
        else:
            # Legacy single-reward path
            single_reward = f1_score - lambda1 * num_retrievals - self.lambda2 * total_latency
            per_state_rewards = [single_reward] * len(states)
            mean_reward = single_reward

        # Update baseline with mean reward
        self.baseline = (
            self.baseline_momentum * self.baseline +
            (1 - self.baseline_momentum) * mean_reward
        )

        # Empty episode: nothing to backprop through.
        if not states:
            self.training_step += 1
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "entropy": 0.0,
                "reward": mean_reward,
                "f1_score": f1_score,
                "num_retrievals": num_retrievals,
                "lambda1": lambda1,
                "baseline": self.baseline,
                "advantage": 0.0,
                "skipped_empty_episode": True,
            }

        # Compute per-state advantages
        advantages = torch.tensor(
            [r - self.baseline for r in per_state_rewards],
            dtype=torch.float32,
            device=self.device,
        )

        # Compute policy loss (REINFORCE) and entropy via forward passes
        log_probs = []
        entropies = []
        for state, action in zip(states, actions):
            log_prob, ent = self.policy.compute_log_prob_and_entropy(state, action)
            log_probs.append(log_prob)
            entropies.append(ent)

        log_probs = torch.stack(log_probs)
        entropy = torch.mean(torch.stack(entropies))

        # REINFORCE loss: -mean( log pi(a|s) * advantage_per_state )
        policy_loss = -(log_probs * advantages).mean()

        # Total loss (entropy bonus is graph-connected for proper gradient flow)
        loss = policy_loss - self.entropy_coef * entropy

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_step += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "reward": mean_reward,
            "f1_score": f1_score,
            "num_retrievals": num_retrievals,
            "lambda1": lambda1,
            "baseline": self.baseline,
            "advantage": float(advantages.mean().item()),
        }
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save training checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "baseline": self.baseline,
            "training_step": self.training_step,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        from realm_retrieve.checkpoint_utils import load_checkpoint as _load, validate_checkpoint_embedding_dim
        checkpoint = _load(path, map_location=self.device)
        validate_checkpoint_embedding_dim(checkpoint, self.policy.embedding_dim)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.baseline = checkpoint["baseline"]
        self.training_step = checkpoint["training_step"]
        return checkpoint["epoch"]

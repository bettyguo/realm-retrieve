#!/usr/bin/env python3
"""
Train Reasoning Step Segmentation Model

Trains a 3-layer transformer classifier to detect reasoning step boundaries.
Dataset: 2,847 human-annotated reasoning traces from DeepSeek-R1
Target: 94.2% F1 on held-out test set

Usage:
    python train_segmentation.py configs/experiments/train_segmentation.yaml
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import wandb

from realm_retrieve.models.segmentation import StepBoundaryClassifier


class SegmentationDataset(Dataset):
    """Dataset for reasoning step boundary detection."""
    
    def __init__(self, data_path: str, tokenizer, window_size: int = 128):
        self.tokenizer = tokenizer
        self.window_size = window_size
        
        # Load annotated data
        self.samples = self._load_data(data_path)
    
    def _load_data(self, data_path: str):
        """Load human-annotated reasoning traces."""
        import json
        
        samples = []
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(sample['text'], add_special_tokens=True)
        labels = sample['boundary_labels']  # [0, 1, 0, 0, 1, ...]
        
        # Pad/truncate to window_size
        if len(tokens) < self.window_size:
            padding = [self.tokenizer.pad_token_id] * (self.window_size - len(tokens))
            tokens = tokens + padding
            labels = labels + [0] * (self.window_size - len(labels))
        else:
            tokens = tokens[:self.window_size]
            labels = labels[:self.window_size]
        
        return {
            'input_ids': torch.tensor(tokens),
            'labels': torch.tensor(labels),
            'attention_mask': torch.ones(self.window_size),
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        logits = model(input_ids, attention_mask)  # [batch, seq_len, 2]
        
        # Reshape for loss
        logits = logits.view(-1, 2)
        labels = labels.view(-1)
        
        # Loss
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
    }


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    tp, fp, fn = 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            logits = logits.view(-1, 2)
            labels = labels.view(-1)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # F1 computation
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


@hydra.main(version_base=None, config_path="configs/experiments", config_name="train_segmentation")
def main(cfg: DictConfig):
    """Main training function."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize W&B
    if cfg.logging.use_wandb:
        wandb.init(project="realm-retrieve", name="segmentation", config=dict(cfg))
    
    # Model
    model = StepBoundaryClassifier(
        base_model=cfg.model.base_model,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data
    train_dataset = SegmentationDataset(
        cfg.data.train_path,
        model.tokenizer,
        window_size=cfg.data.window_size,
    )
    val_dataset = SegmentationDataset(
        cfg.data.val_path,
        model.tokenizer,
        window_size=cfg.data.window_size,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
    )
    
    # Loss (weighted cross-entropy to handle class imbalance)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device))
    
    # Training loop
    best_f1 = 0.0
    for epoch in range(cfg.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Log to W&B
        if cfg.logging.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_metrics['loss'],
                "train_accuracy": train_metrics['accuracy'],
                "val_loss": val_metrics['loss'],
                "val_f1": val_metrics['f1'],
                "val_precision": val_metrics['precision'],
                "val_recall": val_metrics['recall'],
            })
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1,
            }, f"{cfg.training.checkpoint_dir}/best_model.pt")
            print(f"Saved new best model (F1: {best_f1:.4f})")
    
    print(f"\nTraining complete! Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()

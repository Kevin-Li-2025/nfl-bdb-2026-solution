"""
NFL Big Data Bowl 2026 - Optimized Training Script
Implements RMSE loss, GroupKFold, and proper training loop.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, Optional

from model import OptimizedSTGraphTransformer
from dataset import NFLTrackingDataset, create_dummy_data

# --- Configuration ---
CONFIG = {
    # Data
    "data_dir": "./data",
    "t_obs": 10,
    "t_pred": 50,
    
    # Model
    "input_dim": 10,
    "hidden_dim": 256,
    "nhead": 8,
    "num_temporal_layers": 2,
    "num_spatial_layers": 4,
    "dropout": 0.1,
    
    # Training
    "batch_size": 32,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    
    # Validation
    "n_folds": 5,
    "val_fold": 0,
    
    # Output
    "output_dir": "./checkpoints",
    "log_interval": 10,
    "save_best": True,
}


class RMSELoss(nn.Module):
    """Root Mean Squared Error Loss - matches competition metric."""
    def __init__(self, smoothness_weight: float = 0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: [B, N, T, 2] predicted (x, y) coordinates
            target: [B, N, T, 2] ground truth coordinates
            mask: [B, N, T] optional mask for valid timesteps
        """
        # MSE per coordinate
        mse = (pred - target) ** 2
        
        if mask is not None:
            mse = mse * mask.unsqueeze(-1)
            n_valid = mask.sum() * 2  # 2 for x, y
            mse_mean = mse.sum() / (n_valid + 1e-8)
        else:
            mse_mean = mse.mean()
        
        rmse = torch.sqrt(mse_mean + 1e-8)
        
        # Smoothness penalty (velocity consistency)
        if self.smoothness_weight > 0:
            velocity_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            velocity_target = target[:, :, 1:, :] - target[:, :, :-1, :]
            smoothness_loss = ((velocity_pred - velocity_target) ** 2).mean()
            rmse = rmse + self.smoothness_weight * torch.sqrt(smoothness_loss + 1e-8)
        
        return rmse


class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = OptimizedSTGraphTransformer(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            t_obs=config["t_obs"],
            t_pred=config["t_pred"],
            nhead=config["nhead"],
            num_temporal_layers=config["num_temporal_layers"],
            num_spatial_layers=config["num_spatial_layers"],
            dropout=config["dropout"]
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Loss function
        self.criterion = RMSELoss(smoothness_weight=0.1)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
        
        # Scheduler (Cosine Annealing)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Best metric tracking
        self.best_val_rmse = float('inf')
        self.training_history = []

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            player_tracks = batch['player_tracks'].to(self.device)
            agent_types = batch['agent_types'].to(self.device)
            last_positions = batch['last_positions'].to(self.device)
            landing_point = batch['landing_point'].to(self.device)
            ball_track = batch['ball_track'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            preds = self.model(
                player_tracks, agent_types, last_positions,
                landing_point, ball_track
            )
            
            # Loss
            loss = self.criterion(preds, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["grad_clip"]
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if batch_idx % self.config["log_interval"] == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        return total_loss / n_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for batch in tqdm(val_loader, desc="Validating"):
            player_tracks = batch['player_tracks'].to(self.device)
            agent_types = batch['agent_types'].to(self.device)
            last_positions = batch['last_positions'].to(self.device)
            landing_point = batch['landing_point'].to(self.device)
            ball_track = batch['ball_track'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            preds = self.model(
                player_tracks, agent_types, last_positions,
                landing_point, ball_track
            )
            
            loss = self.criterion(preds, targets)
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches

    def save_checkpoint(self, epoch: int, val_rmse: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_rmse': val_rmse,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.output_dir / 'latest.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pth')
            print(f"  ✓ New best model saved (RMSE: {val_rmse:.4f})")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"\n{'='*60}")
        print(f"Starting training for {self.config['epochs']} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_rmse = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log
            print(f"Epoch {epoch+1}/{self.config['epochs']} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val RMSE: {val_rmse:.4f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save
            is_best = val_rmse < self.best_val_rmse
            if is_best:
                self.best_val_rmse = val_rmse
            
            if self.config["save_best"]:
                self.save_checkpoint(epoch, val_rmse, is_best)
            
            # History
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_rmse': val_rmse
            })
        
        # Save training history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training complete. Best Val RMSE: {self.best_val_rmse:.4f}")
        print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--test', action='store_true', help='Run with dummy data')
    args = parser.parse_args()
    
    # Update config
    CONFIG['data_dir'] = args.data_dir
    CONFIG['epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch_size
    CONFIG['lr'] = args.lr
    CONFIG['val_fold'] = args.fold
    
    # Create dummy data if testing
    if args.test:
        create_dummy_data('./data/test', n_plays=50)
        CONFIG['data_dir'] = './data/test'
        CONFIG['epochs'] = 2
    
    # Create datasets
    train_dataset = NFLTrackingDataset(
        CONFIG['data_dir'],
        t_obs=CONFIG['t_obs'],
        t_pred=CONFIG['t_pred'],
        augment=True,
        mode='train',
        fold=CONFIG['val_fold'],
        n_folds=CONFIG['n_folds']
    )
    
    val_dataset = NFLTrackingDataset(
        CONFIG['data_dir'],
        t_obs=CONFIG['t_obs'],
        t_pred=CONFIG['t_pred'],
        augment=False,
        mode='val',
        fold=CONFIG['val_fold'],
        n_folds=CONFIG['n_folds']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Train
    trainer = Trainer(CONFIG)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()

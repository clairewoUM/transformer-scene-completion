"""
train.py - Training script for pose-free scene completion

Usage:
    python train.py --config config.yaml
    python train.py --data_type synthetic --epochs 50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import wandb
from datetime import datetime

from model import PoseFreeSceneCompletion, LightweightSceneCompletion
from dataset import create_dataloaders


class SceneCompletionTrainer:
    """
    Trainer class for scene completion
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = Path(config['output_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize model
        self.model = self.create_model()
        self.model = self.model.to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        # Create dataloaders
        self.train_loader, self.val_loader = create_dataloaders(config)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Logging
        if config.get('use_wandb', False):
            wandb.init(project='scene-completion', config=config)
        
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def create_model(self):
        """Create model based on config"""
        model_type = self.config.get('model_type', 'lightweight')
        
        if model_type == 'lightweight':
            model = LightweightSceneCompletion(
                voxel_resolution=self.config['voxel_resolution']
            )
        else:
            model = PoseFreeSceneCompletion(
                img_size=self.config['img_size'],
                voxel_resolution=self.config['voxel_resolution'],
                embed_dim=self.config.get('embed_dim', 512),
                encoder_depth=self.config.get('encoder_depth', 8),
                decoder_depth=self.config.get('decoder_depth', 4),
                num_heads=self.config.get('num_heads', 8)
            )
        
        return model
    
    def compute_loss(self, pred_occupancy, pred_sdf, batch):
        """
        Compute loss
        
        Components:
        1. Occupancy loss (BCE)
        2. SDF loss (L1)
        3. Smoothness regularization
        """
        losses = {}
        
        if 'gt_voxels' in batch:
            gt_occupancy = batch['gt_voxels'].to(self.device)
            losses['occupancy'] = nn.functional.binary_cross_entropy(
                pred_occupancy, gt_occupancy
            )
        
        if 'gt_sdf' in batch:
            gt_sdf = batch['gt_sdf'].to(self.device)
            losses['sdf'] = nn.functional.l1_loss(pred_sdf, gt_sdf)
        
        # Smoothness loss (total variation)
        tv_loss = self.total_variation_loss(pred_occupancy)
        losses['smoothness'] = tv_loss
        
        # Total loss
        total_loss = 0
        if 'occupancy' in losses:
            total_loss += losses['occupancy']
        if 'sdf' in losses:
            total_loss += 0.5 * losses['sdf']
        total_loss += 0.01 * losses['smoothness']
        
        losses['total'] = total_loss
        
        return losses
    
    @staticmethod
    def total_variation_loss(x):
        """Smoothness regularization"""
        diff_i = torch.abs(x[:, 1:, :, :] - x[:, :-1, :, :])
        diff_j = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_k = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return diff_i.mean() + diff_j.mean() + diff_k.mean()
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        epoch_losses = {'total': 0, 'occupancy': 0, 'sdf': 0, 'smoothness': 0}
        
        for batch in pbar:
            video = batch['video'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                pred_occupancy, pred_sdf = self.model(video)
                losses = self.compute_loss(pred_occupancy, pred_sdf, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(losses['total']).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'occ': losses.get('occupancy', 0).item() if 'occupancy' in losses else 0
            })
            
            self.global_step += 1
            
            # Log to wandb
            if self.config.get('use_wandb', False) and self.global_step % 10 == 0:
                wandb.log({f'train/{k}': v.item() if hasattr(v, 'item') else v 
                          for k, v in losses.items()}, step=self.global_step)
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    def validate(self):
        """Validation"""
        self.model.eval()
        
        val_losses = {'total': 0, 'occupancy': 0, 'sdf': 0, 'smoothness': 0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                video = batch['video'].to(self.device)
                
                pred_occupancy, pred_sdf = self.model(video)
                losses = self.compute_loss(pred_occupancy, pred_sdf, batch)
                
                for key in val_losses:
                    if key in losses:
                        val_losses[key] += losses[key].item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)
        
        return val_losses
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        # Save latest
        checkpoint_path = self.output_dir / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = self.output_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(self.config['epochs']):
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Print results
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss:   {val_losses['total']:.4f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    **{f'val/{k}': v for k, v in val_losses.items()}
                }, step=self.global_step)
            
            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            
            if epoch % self.config.get('save_freq', 10) == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
        
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train scene completion model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_type', type=str, default='synthetic', 
                       choices=['synthetic', 'video', 'realsense'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--model_type', type=str, default='lightweight',
                       choices=['lightweight', 'full'])
    parser.add_argument('--use_wandb', action='store_true')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'data_type': args.data_type,
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'model_type': args.model_type,
            'use_wandb': args.use_wandb,
            
            # Model parameters
            'num_frames': 10,
            'img_size': 224,
            'voxel_resolution': 64,
            'embed_dim': 512,
            'encoder_depth': 8,
            'decoder_depth': 4,
            'num_heads': 8,
            
            # Training parameters
            'weight_decay': 0.01,
            'num_workers': 4,
            'save_freq': 10,
        }
    
    # Create trainer and train
    trainer = SceneCompletionTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
"""
Training utilities for contrastive encoder and cross-encoder.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List

from .models import ContrastiveEncoder, CrossEncoder, ContrastiveLoss, TripletLoss


class ContrastiveTrainer:
    """Trainer for contrastive encoder."""
    
    def __init__(
        self,
        model: ContrastiveEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float,
        num_epochs: int,
        temperature: float,
        margin: float,
        gradient_clip_max_norm: float,
        output_dir: Path
    ):
        """
        Initialize contrastive trainer.
        
        Args:
            model: Contrastive encoder model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Computation device
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            temperature: Temperature for contrastive loss
            margin: Margin for triplet loss
            gradient_clip_max_norm: Max norm for gradient clipping
            output_dir: Directory to save outputs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.output_dir = output_dir
        
        # Loss functions
        self.contrastive_loss_fn = ContrastiveLoss(temperature=temperature)
        self.triplet_loss_fn = TripletLoss(margin=margin)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            anchor_ids = batch['anchor_input_ids'].to(self.device)
            anchor_mask = batch['anchor_attention_mask'].to(self.device)
            positive_ids = batch['positive_input_ids'].to(self.device)
            positive_mask = batch['positive_attention_mask'].to(self.device)
            negative_ids = batch['negative_input_ids'].to(self.device)
            negative_mask = batch['negative_attention_mask'].to(self.device)
            
            # Forward pass
            anchor_emb = self.model(anchor_ids, anchor_mask)
            positive_emb = self.model(positive_ids, positive_mask)
            negative_emb = self.model(negative_ids, negative_mask)
            
            # Compute losses
            loss_contrastive = self.contrastive_loss_fn(anchor_emb, positive_emb, negative_emb)
            loss_triplet = self.triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
            loss = loss_contrastive + 0.5 * loss_triplet
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.gradient_clip_max_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                embeddings = self.model(input_ids, attention_mask)
                norms = torch.norm(embeddings, p=2, dim=1)
                total_loss += torch.abs(norms - 1.0).mean().item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Train the model for all epochs."""
        print("\n" + "="*80)
        print("TRAINING CONTRASTIVE ENCODER")
        print("="*80)
        
        for epoch in range(self.num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*80}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            print(f"\nTrain Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            print(f"Val Loss (norm deviation): {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)
                print("Best model saved!")
        
        # Plot training history
        self.plot_history()
    
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        filename = 'best_contrastive_encoder.pt' if is_best else f'contrastive_encoder_epoch_{epoch}.pt'
        torch.save(checkpoint, self.output_dir / filename)
    
    def plot_history(self):
        """Plot training history."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.train_losses, label='Train Loss', marker='o')
        ax.plot(self.val_losses, label='Val Loss', marker='s')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title('Training History - Contrastive Encoder', fontweight='bold', pad=20)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.savefig(self.output_dir / 'contrastive_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()


class CrossEncoderTrainer:
    """Trainer for cross-encoder."""
    
    def __init__(
        self,
        model: CrossEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float,
        num_epochs: int,
        gradient_clip_max_norm: float,
        output_dir: Path
    ):
        """
        Initialize cross-encoder trainer.
        
        Args:
            model: Cross-encoder model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Computation device
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            gradient_clip_max_norm: Max norm for gradient clipping
            output_dir: Directory to save outputs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.output_dir = output_dir
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            predictions = self.model(input_ids, attention_mask)
            loss = self.criterion(predictions, labels.squeeze(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip_max_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                predictions = self.model(input_ids, attention_mask)
                loss = self.criterion(predictions, labels.squeeze(-1))
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Train the model for all epochs."""
        print("\n" + "="*80)
        print("TRAINING CROSS-ENCODER")
        print("="*80)
        
        for epoch in range(self.num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*80}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            print(f"\nTrain Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)
                print("Best model saved!")
        
        # Plot training history
        self.plot_history()
    
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        filename = 'best_cross_encoder.pt' if is_best else f'cross_encoder_epoch_{epoch}.pt'
        torch.save(checkpoint, self.output_dir / filename)
    
    def plot_history(self):
        """Plot training history."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.train_losses, label='Train Loss', marker='o')
        ax.plot(self.val_losses, label='Val Loss', marker='s')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title('Training History - Cross-Encoder', fontweight='bold', pad=20)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.savefig(self.output_dir / 'cross_encoder_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

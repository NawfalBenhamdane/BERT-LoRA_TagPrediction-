"""
Neural network models for contrastive learning and cross-encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ContrastiveEncoder(nn.Module):
    """Contrastive encoder for learning exercise embeddings."""
    
    def __init__(
        self,
        base_model_name: str,
        embedding_dim: int,
        projection_dim: int,
        normalize: bool = True
    ):
        """
        Initialize contrastive encoder.
        
        Args:
            base_model_name: Name of the pretrained model
            embedding_dim: Dimension of base model embeddings
            projection_dim: Dimension of projected embeddings
            normalize: Whether to normalize embeddings
        """
        super().__init__()
        self.normalize = normalize
        
        # Base encoder
        self.encoder = AutoModel.from_pretrained(base_model_name)
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Projected embeddings
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        embeddings = self.projector(pooled_output)
        
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class CrossEncoder(nn.Module):
    """Cross-encoder for scoring (exercise, tag) pairs."""
    
    def __init__(
        self,
        base_model_name: str,
        embedding_dim: int,
        hidden_dim: int
    ):
        """
        Initialize cross-encoder.
        
        Args:
            base_model_name: Name of the pretrained model
            embedding_dim: Dimension of base model embeddings
            hidden_dim: Hidden dimension for classifier
        """
        super().__init__()
        
        # Base encoder
        self.encoder = AutoModel.from_pretrained(base_model_name)
        
        # Binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Relevance scores
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        score = self.classifier(pooled_output)
        return score.squeeze(-1)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for triplet learning."""
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for scaling
        """
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings
            negative: Negative embeddings
            
        Returns:
            Loss value
        """
        batch_size = anchor.size(0)
        
        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature
        
        # Stack logits
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        loss = self.criterion(logits, labels)
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with margin."""
    
    def __init__(self, margin: float = 0.5):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings
            negative: Negative embeddings
            
        Returns:
            Loss value
        """
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Challenge CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""

import torch.nn as nn
import torch
from model.vit import TransformerEncoder, get_positional_embeddings


__all__ = ["Challenge"]


class Challenge(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()
        
        dropout_rate = 0.2

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        
        self.embedding_dim = 32
        self.feature_to_token = nn.Linear(128, self.embedding_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        
        self.register_buffer("pos_embed", get_positional_embeddings(5, self.embedding_dim).detach().float().unsqueeze(0))
        
        self.transformer_blocks = nn.ModuleList([TransformerEncoder(self.embedding_dim, 2) for i in range(2)])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embedding_dim, 2),
        )
    
    def forward(self, x):
        features = self.features(x)
        
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.feature_to_token(tokens)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed
        
        for block in self.transformer_blocks:
            tokens = block(tokens)

        return self.classifier(tokens[:, 0])

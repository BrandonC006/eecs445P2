"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Challenge CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""

import torch.nn as nn


__all__ = ["Challenge"]


class Challenge(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()
        
        dropout_rate = 0.3

        self.features = nn.Sequential(
            nn.Conv2D(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2),
            
            nn.Conv2D(32, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2),
            
            nn.Conv2D(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AveragePool2D((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

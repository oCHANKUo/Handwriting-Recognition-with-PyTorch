# Convolutional Neural Network

import torch
from torch import nn

# defining a neural network called ImageClassifier
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # Output: 26x26
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3), # Output: 24x24
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), # Output: 22x22
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)       # Fully connected layer for 10 classes
        )

    def forward(self, x):
        return self.model(x)
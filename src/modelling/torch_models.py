"""
torch_models.py

This module defines the GBMPurity class, a PyTorch neural network model for predicting GBM purity.
The model consists of two hidden layers with dropout for regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP2h(nn.Module):
    def __init__(self, input_size, h1=32, h2=16, p_dropout=0.4):
        """
        Initialize the GBMPurity model.

        Parameters:
            input_size (int): Number of input features.
            h1 (int, optional): Number of neurons in the first hidden layer. Default is 32.
            h2 (int, optional): Number of neurons in the second hidden layer. Default is 16.
            p_dropout (float, optional): Dropout probability. Default is 0.4.
        """
        super(MLP2h, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, 1)
        self.dropout = nn.Dropout(p_dropout)
        
    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        return x

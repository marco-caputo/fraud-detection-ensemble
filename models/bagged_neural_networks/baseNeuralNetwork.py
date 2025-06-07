import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np

class baseNeuralNetwork(nn.Module):
    """
    A single neural network model used as a base learner in the bagging ensemble.
    It takes a 10-dimensional latent representation as input and outputs a
    single logit for binary classification (e.g., Fraud/Legit).
    """
    def __init__(self, input_dim=10, hidden_dims=[64, 32], dropout_rate=0.3):
        super(baseNeuralNetwork, self).__init__()

        layers = []
        current_dim = input_dim

        # Create hidden layers with ReLU activation and Dropout
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        # Output layer for binary classification (single logit)
        layers.append(nn.Linear(current_dim, 1))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
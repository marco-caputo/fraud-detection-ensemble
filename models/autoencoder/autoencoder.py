import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import AUTOENC_HIDDEN_LAYER_SIZE, LATENT_DIMENSION, N_FEATURES

class Autoencoder(nn.Module):
    def __init__(self, input_dim=N_FEATURES, hidden_dim=AUTOENC_HIDDEN_LAYER_SIZE, latent_dim=LATENT_DIMENSION):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

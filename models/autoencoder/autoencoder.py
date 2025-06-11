import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int, latent_dim: int):

        super(Autoencoder, self).__init__()

        encode_modules = ([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)] + \
                ([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)] * (hidden_layers-1)) +
                [nn.Linear(hidden_dim, latent_dim), nn.ReLU(), nn.BatchNorm1d(latent_dim)]) if hidden_layers > 0 else \
                [nn.Linear(input_dim, latent_dim), nn.ReLU(), nn.BatchNorm1d(latent_dim)]

        decode_modules = ([nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)] +
                ([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)] * (hidden_layers-1)) + \
                [nn.Linear(hidden_dim, input_dim), nn.Sigmoid()]) if hidden_layers > 0 else \
                [nn.Linear(latent_dim, input_dim), nn.Sigmoid()]

        self.encoder = nn.Sequential(*encode_modules)  # Encoder
        self.decoder = nn.Sequential(*decode_modules)  # Decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

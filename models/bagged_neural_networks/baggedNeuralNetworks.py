import baseNeuralNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
class BaggedNeuralNetworks(nn.Module):
    """
    An ensemble of Neural Networks trained using bagging.
    Each network is trained on a bootstrapped sample of the training data.
    """
    def __init__(self, n_estimators=5, input_dim=10, hidden_dims=[64, 32], dropout_rate=0.3):
        """
        Initializes the Bagged Neural Networks ensemble.

        Args:
            n_estimators (int): The number of individual neural networks in the ensemble.
            input_dim (int): Input dimensionality for each neural network.
            hidden_dims (list): Hidden layer dimensions for each neural network.
            dropout_rate (float): Dropout rate for each neural network.
        """
        super(BaggedNeuralNetworks, self).__init__()
        self.n_estimators = n_estimators
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Create a ModuleList to hold individual neural networks
        # ModuleList is important for PyTorch to correctly register parameters
        self.models = nn.ModuleList([
            baseNeuralNetwork(input_dim, hidden_dims, dropout_rate)
            for _ in range(n_estimators)
        ])

    def fit(self, X_latent, y, epochs=50, batch_size=32, lr=0.001):
        """
        Trains each neural network in the ensemble using bootstrapped samples.

        Args:
            X_latent (torch.Tensor): The input latent representations (e.g., from Autoencoder).
                                     Shape: (num_samples, input_dim).
            y (torch.Tensor): The corresponding labels (0 or 1). Shape: (num_samples,).
            epochs (int): Number of training epochs for each base learner.
            batch_size (int): Batch size for training each base learner.
            lr (float): Learning rate for the optimizer of each base learner.
        """
        num_samples = X_latent.shape[0]
        dataset = TensorDataset(X_latent, y)
        criterion = nn.BCEWithLogitsLoss() # Use BCEWithLogitsLoss for single logit output and binary classification

        print(f"Training {self.n_estimators} individual neural networks...")

        for i, model in enumerate(self.models):
            print(f"\nTraining Model {i+1}/{self.n_estimators}...")

            # Create a bootstrapped dataset (sampling with replacement)
            # This is a key aspect of bagging
            bootstrap_indices = np.random.choice(num_samples, num_samples, replace=True)
            bootstrapped_subset = Subset(dataset, bootstrap_indices)
            bootstrapped_loader = DataLoader(bootstrapped_subset, batch_size=batch_size, shuffle=True)

            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Set model to training mode
            model.train()

            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in bootstrapped_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y.float()) # .squeeze() to remove singleton dimension, .float() for BCEWithLogitsLoss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(bootstrapped_loader):.4f}")

            # Set model back to evaluation mode after training
            model.eval()
        print("\nAll models trained.")

    def forward(self, X_latent):
        """
        Aggregates predictions from all individual neural networks.
        Inference mode for the ensemble.

        Args:
            X_latent (torch.Tensor): The input latent representations for prediction.
                                     Shape: (num_samples_to_predict, input_dim).

        Returns:
            torch.Tensor: The averaged logits from all models.
                          Shape: (num_samples_to_predict, 1).
                          These can then be passed to a sigmoid to get probabilities
                          or directly used by the Voting Aggregator.
        """
        # Ensure models are in evaluation mode during inference
        for model in self.models:
            model.eval()

        # Store predictions from each model
        all_predictions = []

        # No gradient calculation needed during inference
        with torch.no_grad():
            for model in self.models:
                # Get raw logits from each model
                predictions = model(X_latent)
                all_predictions.append(predictions)

        # Stack predictions and average them
        # (num_estimators, num_samples_to_predict, 1) -> (num_samples_to_predict, 1)
        # The 'Voting Aggregator' can then apply a threshold or majority vote
        # on these averaged logits/probabilities after sigmoid.
        averaged_predictions = torch.mean(torch.stack(all_predictions), dim=0)
        return averaged_predictions

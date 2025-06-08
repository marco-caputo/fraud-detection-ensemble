import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np

from models.bagged_neural_networks.base_neural_network import BaseNeuralNetwork


class BaggedNeuralNetworks(nn.Module):
    """
    An ensemble of Neural Networks trained using bagging.
    Each network is trained on a bootstrapped sample of the training data.
    """
    def __init__(self, n_estimators: int, input_dim: int, hidden_dims: list[int], dropout_rate: float, device):
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
        self.device = device

        # Create a ModuleList to hold individual neural networks
        # ModuleList is important for PyTorch to correctly register parameters
        self.models = nn.ModuleList([
            BaseNeuralNetwork(input_dim, hidden_dims, dropout_rate)
            for _ in range(n_estimators)
        ])

    def fit(self, X_latent, y, epochs: int, batch_size: int, lr: float, val_loader: DataLoader,
             patience: int = 10, min_delta: float = 1e-4):
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

            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            for epoch in range(epochs):
                # Set model to training mode
                model.train()

                total_loss = 0
                for batch_X, batch_y in bootstrapped_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y.float()) # .squeeze() to remove singleton dimension, .float() for BCEWithLogitsLoss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # Validation phase
                val_loss = self.evaluate_neural_network(model, val_loader)
                print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {total_loss / len(bootstrapped_loader):.6f}, Val Loss: {val_loss:.6f}")

                # Early stopping check
                if val_loss + min_delta < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()  # Save best model
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}. Best Val Loss: {best_val_loss:.6f}")
                        break

            # Restore best model weights
            if best_model_state:
                model.load_state_dict(best_model_state)
            # Set model back to evaluation mode after training
            model.eval()
        print("\nAll models trained.")

    def evaluate_neural_network(self, model: BaseNeuralNetwork, val_loader: DataLoader) -> float:
        model.eval()
        total_loss = 0.0
        total_samples = 0

        loss_fn = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = model(batch_X)  # logits
                loss = loss_fn(outputs, batch_y.float())  # Call loss_fn here
                total_loss += loss.item() * batch_X.size(0)
                total_samples += batch_X.size(0)

        return total_loss / total_samples

    def forward(self, X_latent, batch_size=512):
        """
        Aggregates predictions from all individual neural networks using batch-wise evaluation.
        Returns averaged logits (before sigmoid).
        """
        dataset = DataLoader(X_latent, batch_size=batch_size)
        all_model_outputs = []

        for model in self.models:
            model.eval()
            predictions = []

            with torch.no_grad():
                for batch_X in dataset:
                    batch_X = batch_X.to(self.device)
                    output = model(batch_X)
                    predictions.append(output.detach().cpu())  # Detach and move to CPU

            predictions = torch.cat(predictions, dim=0)  # (num_samples, 1)
            all_model_outputs.append(predictions)

        # Stack predictions and average them
        # (num_estimators, num_samples_to_predict, 1) -> (num_samples_to_predict, 1)
        # The 'Voting Aggregator' can then apply a threshold or majority vote
        # on these averaged logits/probabilities after sigmoid.
        all_model_outputs = torch.stack(all_model_outputs, dim=0)  # (n_estimators, num_samples, 1)
        averaged_predictions = torch.mean(all_model_outputs, dim=0)  # (num_samples, 1)
        return averaged_predictions

import os
import sys

import torch
import pandas as pd
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.autoencoder.autoencoder import Autoencoder
from plotting.line_plots import save_training_error_plot
from config import (RANDOM_SEED, AUTOENC_TEST_SIZE, AUTENC_VALIDATION_SIZE, AUTOENC_LEARNING_RATE, AUTOENC_EPOCHS,
                    AUTOENC_BATCH_SIZE, DATASET_FOLDER_NAME, CLEANED_DATASET_NAME, AUTOENC_STATE_DICT_FILENAME,
                    AUTOENC_HIDDEN_LAYER_SIZE, AUTOENC_HIDDEN_LAYERS, LATENT_DIMENSION, N_FEATURES)


def train_autoencoder(model: Autoencoder, train_loader: DataLoader, val_loader: DataLoader,
                      criterion, optimizer: Optimizer, device, epochs=AUTOENC_EPOCHS) -> tuple[
    list[float], list[float]]:
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)  # Move data to the device (GPU or CPU)
            optimizer.zero_grad()  # Reset gradients
            output = model(x)  # Forward pass through the encoder and decoder
            loss = criterion(output, x)  # Compute loss in reconstructing the input
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            train_loss += loss.item() * x.size(0)  # Accumulate loss proportional to batch size

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        val_loss = evaluate_autoencoder(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses


def evaluate_autoencoder(model: Autoencoder, test_loader: DataLoader, criterion, device) -> float:
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            output = model(x)
            loss = criterion(output, x)
            test_loss += loss.item() * x.size(0)

    return test_loss / len(test_loader.dataset)


def train_autoencoder_model(model_name: str = "Autoencoder",
                            batch_size: int = AUTOENC_BATCH_SIZE,
                            learning_rate: float = AUTOENC_LEARNING_RATE,
                            epochs: int = AUTOENC_EPOCHS,
                            hidden_layer_size: int = AUTOENC_HIDDEN_LAYER_SIZE,
                            hidden_layers: int = AUTOENC_HIDDEN_LAYERS,
                            ) -> tuple[Autoencoder, float]:
    """ Train an autoencoder model on the cleaned dataset and return the trained model and test loss."""

    # All features except the last column (target)
    X = pd.read_csv(f"../../{DATASET_FOLDER_NAME}/{CLEANED_DATASET_NAME}.csv").iloc[:, :-1]

    # Convert to torch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    # Train/test split
    X_train, X_test = train_test_split(X_tensor, test_size=AUTOENC_TEST_SIZE, random_state=RANDOM_SEED)
    X_train, X_validation = train_test_split(X_train, test_size=AUTENC_VALIDATION_SIZE, random_state=RANDOM_SEED)

    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(TensorDataset(X_validation), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test), batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=N_FEATURES,
                        hidden_dim=hidden_layer_size,
                        hidden_layers=hidden_layers,
                        latent_dim=LATENT_DIMENSION).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training and evaluation functions for the autoencoder
    print("Starting training of the autoencoder...")
    train_losses, val_losses = train_autoencoder(model, train_loader, validation_loader, criterion, optimizer, device,
                                                 epochs)
    print("Training completed.")
    save_training_error_plot(model_name, train_losses, val_losses)

    print("Starting evaluation of the autoencoder on the test set...")
    test_loss = evaluate_autoencoder(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")

    return model, test_loss


if __name__ == "__main__":
    model, test_loss = train_autoencoder_model()
    print("Evaluation completed. Saving the model state dictionary.")
    torch.save(model.state_dict(), f"{AUTOENC_STATE_DICT_FILENAME}.pth")

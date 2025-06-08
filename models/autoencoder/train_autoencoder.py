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
from config import *

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))


def train_autoencoder(model: Autoencoder, train_loader: DataLoader, val_loader: DataLoader,
                      criterion, optimizer: Optimizer, device, epochs=AUTOENC_EPOCHS,
                      patience: int = 10, min_delta: float = 1e-4) -> tuple[list[float], list[float]]:
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        val_loss = evaluate_autoencoder(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

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

    train_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{TRAIN_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")
    validation_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{VALIDATION_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")
    test_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{TEST_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")

    # All features except the last column (target)
    X_train = pd.read_csv(train_path).iloc[:, :-1]
    X_validation = pd.read_csv(validation_path).iloc[:, :-1]
    X_test = pd.read_csv(test_path).iloc[:, :-1]

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_validation_tensor = torch.tensor(X_validation.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(TensorDataset(X_validation_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=batch_size, shuffle=False)

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
    torch.save(model.state_dict(), os.path.join(script_dir, f"{AUTOENC_STATE_DICT_FILENAME}.pth"))

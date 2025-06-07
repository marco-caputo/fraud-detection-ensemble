import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import *

# Import the custom classes
from baseNeuralNetwork import baseNeuralNetwork
from baggedNeuralNetworks import BaggedNeuralNetworks

def load_data(file_path: str, input_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load the dataset from a CSV file and return the features and labels as tensors.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        input_dim (int): The number of features in the dataset.

    Returns:
        tuple: A tuple containing two tensors:
            - X (torch.Tensor): The input features of shape (num_samples, input_dim).
            - y (torch.Tensor): The labels of shape (num_samples,).
    """
    data = pd.read_csv(file_path).values  # Load the dataset as a NumPy array
    X = torch.tensor(data[:, :input_dim], dtype=torch.float32)
    y = torch.tensor(data[:, input_dim], dtype=torch.float32).unsqueeze(1)  # Ensure y is a column vector
    return X, y

def train_bagged_neural_networks(
    model_name: str = "BaggedNN",
    n_estimators: int = N_ESTIMATORS,
    input_dim: int = INPUT_DIM,
    hidden_dims: list = HIDDEN_DIMS,
    dropout_rate: float = DROPOUT_RATE,
    epochs_per_model: int = EPOCHS_PER_MODEL,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE
) -> tuple[BaggedNeuralNetworks, float]:
    # 1. Load the dataset
    train_path= os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "..", "..", DATASET_FOLDER_NAME, 
        f"{TRAIN_DATASET_PREFIX}_{ENCODED_DATASET_NAME}.csv"
    )
    validation_path= os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "..", "..", DATASET_FOLDER_NAME, 
        f"{VALIDATION_DATASET_PREFIX}_{ENCODED_DATASET_NAME}.csv"
    )
    X_train, y_train = load_data(train_path, input_dim)
    X_val, y_val = load_data(validation_path, input_dim)
    
    # 2. Initialize the Bagged Neural Networks ensemble
    ensemble_model = BaggedNeuralNetworks(
        n_estimators=n_estimators,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    )

    # 3. Train the ensemble
    ensemble_model.fit(
        X_train, y_train,
        epochs=epochs_per_model,
        batch_size=batch_size,
        lr=learning_rate
    )

    # 4. Evaluate the ensemble
    ensemble_model.eval()  # Set ensemble to evaluation mode
    with torch.no_grad():
        # Get averaged logits from the ensemble
        averaged_logits = ensemble_model(X_val)

        # Convert logits to probabilities using sigmoid
        probabilities = torch.sigmoid(averaged_logits)

        # Convert probabilities to binary predictions (0 or 1)
        predictions = (probabilities > 0.5).float()

        # Calculate accuracy
        correct_predictions = (predictions.squeeze() == y_val).sum().item()
        total_samples = y_val.shape[0]
        accuracy = correct_predictions / total_samples

        print(f"Accuracy on validation data: {accuracy:.4f}")
    return ensemble_model, accuracy

if __name__ == "__main__":
    print("--- Starting Bagged Neural Network Training ---")
    model, accuracy = train_bagged_neural_networks(
        model_name="BaggedNN",
        n_estimators=N_ESTIMATORS,
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        dropout_rate=DROPOUT_RATE,
        epochs_per_model=EPOCHS_PER_MODEL,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    print(f"Trained model: {model}")
    print(f"Validation accuracy: {accuracy:.4f}")
    torch.save(model.state_dict(), f"{BAGGED_NN_MODEL_FILENAME}_state_dict.pth")

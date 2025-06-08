import sys
import torch
import pandas as pd
import os

from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import *

from bagged_neural_networks import BaggedNeuralNetworks

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

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
    data = pd.read_csv(file_path)  # Load the dataset as a NumPy array
    X = torch.tensor(data.iloc[:, :input_dim].values, dtype=torch.float32)
    y = torch.tensor(data.iloc[:, input_dim].values, dtype=torch.float32).unsqueeze(1)  # Ensure y is a column vector
    return X, y

def train_bagged_neural_networks(
    model_name: str = "BaggedNN",
    n_estimators: int = N_ESTIMATORS,
    input_dim: int = INPUT_DIM,
    hidden_dims: list = HIDDEN_DIMS,
    dropout_rate: float = DROPOUT_RATE,
    epochs_per_model: int = EPOCHS_PER_MODEL,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = NN_LEARNING_RATE
) -> tuple[BaggedNeuralNetworks, float]:
    # 1. Load the dataset
    train_path= os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME,
                             f"{TRAIN_DATASET_PREFIX}_{ENCODED_DATASET_NAME}.csv")
    validation_path= os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME,
                                  f"{VALIDATION_DATASET_PREFIX}_{ENCODED_DATASET_NAME}.csv")
    test_path= os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME,
                                  f"{TEST_DATASET_PREFIX}_{ENCODED_DATASET_NAME}.csv")
    X_train, y_train = load_data(train_path, input_dim)
    X_val, y_val = load_data(validation_path, input_dim)
    X_test, y_test = load_data(test_path, input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Initialize the Bagged Neural Networks ensemble
    ensemble_model = BaggedNeuralNetworks(
        n_estimators=n_estimators,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        device=device
    ).to(device)

    # 3. Train the ensemble
    ensemble_model.fit(
        X_train, y_train,
        epochs=epochs_per_model,
        batch_size=batch_size,
        lr=learning_rate,
        val_loader=DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    )

    # 4. Evaluate the ensemble
    ensemble_model.eval()  # Set ensemble to evaluation mode

    def evaluate_in_batches(model, X, y, batch_size:int, device):
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size].to(device)
                batch_y = y[i:i + batch_size]

                logits = model(batch_X)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().squeeze()
                correct += (preds == batch_y.long().squeeze()).sum().item()
                total += batch_y.size(0)
        return correct / total

    accuracy = evaluate_in_batches(ensemble_model, X_test, y_test, batch_size=1024, device=ensemble_model.device)
    print(f"Accuracy on test data: {accuracy:.6f}")
    return ensemble_model, accuracy

if __name__ == "__main__":
    print("Starting Bagged Neural Network Training...")
    model, accuracy = train_bagged_neural_networks(
        model_name="BaggedNN",
        n_estimators=N_ESTIMATORS,
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        dropout_rate=DROPOUT_RATE,
        epochs_per_model=EPOCHS_PER_MODEL,
        batch_size=BATCH_SIZE,
        learning_rate=NN_LEARNING_RATE
    )
    print(f"Trained model: {model}")
    print(f"Validation accuracy: {accuracy:.6f}")
    torch.save(model.state_dict(), os.path.join(script_dir, f"{BAGGED_NN_MODEL_FILENAME}_state_dict.pth"))

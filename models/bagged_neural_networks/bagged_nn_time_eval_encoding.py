import pandas as pd
import torch
import os
import time


from config import *
from bagged_neural_networks import BaggedNeuralNetworks

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cpu")


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

def evaluate_in_batches(model, X, y, batch_size: int, device):
        correct = 0
        total = 0
        start_time = time.time()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size].to(device)
                batch_y = y[i:i + batch_size]

                logits = model(batch_X)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().squeeze()
                correct += (preds == batch_y.long().squeeze()).sum().item()
                total += batch_y.size(0)
        end_time = time.time()
        print(f"Evaluation time: {end_time - start_time:.2f} seconds")
        print(f"Average prediction time per sample: {(end_time - start_time) / total:.6f} seconds")
        return correct / total


nn_ensemble = BaggedNeuralNetworks(
    n_estimators=N_ESTIMATORS,
    input_dim=INPUT_DIM,
    hidden_dims=HIDDEN_DIMS,
    dropout_rate=DROPOUT_RATE,
    device=device
)
nn_ensemble.load_state_dict(torch.load(os.path.join(script_dir, f"{BAGGED_NN_MODEL_FILENAME}_state_dict.pth")))

nn_ensemble_not_encoded = BaggedNeuralNetworks(
    n_estimators=N_ESTIMATORS,
    input_dim=N_FEATURES,
    hidden_dims=HIDDEN_DIMS,
    dropout_rate=DROPOUT_RATE,
    device=device
)
nn_ensemble_not_encoded.load_state_dict(torch.load(os.path.join(script_dir, f"{BAGGED_NN_MODEL_FILENAME}_not_encoded_state_dict.pth")))


test_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{TEST_DATASET_PREFIX}_{ENCODED_DATASET_NAME}.csv")
not_encoded_test_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{TEST_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")
outliers_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{OUTLIER_DATASET_NAME}.csv")
outliers_encoded_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{ENCODED_OUTLIER_DATASET_NAME}.csv")

X_test_encoded, y_test_encoded = load_data(test_path, INPUT_DIM)
X_test_not_encoded, y_test_not_encoded = load_data(not_encoded_test_path, N_FEATURES)
X_outliers_encoded, y_outliers_encoded = load_data(outliers_encoded_path, INPUT_DIM)
X_outliers, y_outliers = load_data(outliers_path, N_FEATURES)


nn_ensemble.eval()
print("Evaluating Bagged Neural Networks on encoded test data...")
accuracy = evaluate_in_batches(nn_ensemble, X_test_encoded, y_test_encoded, batch_size=1024, device=nn_ensemble.device)
print("Evaluating Bagged Neural Networks on encoded outlier data...")
accuracy_outliers_encoded = evaluate_in_batches(nn_ensemble, X_outliers_encoded, y_outliers_encoded,
                                                  batch_size=1024, device=nn_ensemble.device)

print("Evaluating Bagged Neural Networks on not encoded data...")
nn_ensemble_not_encoded.eval()
accuracy_not_encoded = evaluate_in_batches(nn_ensemble_not_encoded, X_test_not_encoded, y_test_not_encoded,
                                           batch_size=1024, device=nn_ensemble_not_encoded.device)
print("Evaluating Bagged Neural Networks on not encoded outlier data...")
accuracy_outliers_not_encoded = evaluate_in_batches(nn_ensemble_not_encoded, X_outliers, y_outliers,
                                                      batch_size=1024, device=nn_ensemble_not_encoded.device)

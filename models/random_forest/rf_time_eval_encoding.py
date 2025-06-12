import joblib
import pandas as pd
import torch
import os
import time


from config import *

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


test_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{TEST_DATASET_PREFIX}_{ENCODED_DATASET_NAME}.csv")
not_encoded_test_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{TEST_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")
outliers_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{OUTLIER_DATASET_NAME}.csv")
outliers_encoded_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{ENCODED_OUTLIER_DATASET_NAME}.csv")

X_test_encoded, y_test_encoded = load_data(test_path, INPUT_DIM)
X_test_not_encoded, y_test_not_encoded = load_data(not_encoded_test_path, N_FEATURES)
X_outliers_encoded, y_outliers_encoded = load_data(outliers_encoded_path, INPUT_DIM)
X_outliers, y_outliers = load_data(outliers_path, N_FEATURES)

rf = joblib.load(os.path.join(script_dir, f"{RANDOM_FOREST_MODEL_FILENAME}.joblib"))
rf_not_encoded = joblib.load(os.path.join(script_dir, f"{RANDOM_FOREST_MODEL_FILENAME}_not_encoded.joblib"))

start_time = time.time()
y_pred = rf.predict(X_test_encoded)
end_time = time.time()
print(f"Evaluation time for encoded data: {end_time - start_time:.2f} seconds")
print(f"Average prediction time per sample for encoded data: {(end_time - start_time) / len(X_test_encoded):.6f} seconds")

# Evaluate on outliers
start_time_outliers = time.time()
y_pred_outliers = rf.predict(X_outliers_encoded)
end_time_outliers = time.time()
print(f"Evaluation time for outliers (encoded data): {end_time_outliers - start_time_outliers:.2f} seconds")
print(f"Average prediction time per sample for outliers (encoded data): {(end_time_outliers - start_time_outliers) / len(X_outliers_encoded):.6f} seconds")

start_time_not_encoded = time.time()
y_pred = rf_not_encoded.predict(X_test_not_encoded)
end_time_not_encoded = time.time()
print(f"Evaluation time for not encoded data: {end_time_not_encoded - start_time_not_encoded:.2f} seconds")
print(f"Average prediction time per sample for not encoded data: {(end_time_not_encoded - start_time_not_encoded) / len(X_test_not_encoded):.6f} seconds")

# Evaluate on outliers not encoded
start_time_outliers_not_encoded = time.time()
y_pred_outliers_not_encoded = rf_not_encoded.predict(X_outliers)
end_time_outliers_not_encoded = time.time()
print(f"Evaluation time for outliers (not encoded data): {end_time_outliers_not_encoded - start_time_outliers_not_encoded:.2f} seconds")
print(f"Average prediction time per sample for outliers (not encoded data): {(end_time_outliers_not_encoded - start_time_outliers_not_encoded) / len(X_outliers):.6f} seconds")

import os
import sys

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import *
from models.autoencoder.autoencoder import Autoencoder

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model
model = Autoencoder(input_dim=N_FEATURES, hidden_dim=AUTOENC_HIDDEN_LAYER_SIZE, hidden_layers=AUTOENC_HIDDEN_LAYERS,
                    latent_dim=LATENT_DIMENSION)
model.load_state_dict(torch.load(os.path.join(script_dir, f"{AUTOENC_STATE_DICT_FILENAME}.pth")))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{TRAIN_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")
validation_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{VALIDATION_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")
test_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{TEST_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")

# Load the dataset
clean_train_dataset = pd.read_csv(train_path)
clean_validation_dataset = pd.read_csv(validation_path)
clean_test_dataset = pd.read_csv(test_path)

# Convert features to torch tensor
X_train_tensor = torch.tensor(clean_train_dataset.iloc[:, :-1].values, dtype=torch.float32).to(device)
X_validation_tensor = torch.tensor(clean_validation_dataset.iloc[:, :-1].values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(clean_test_dataset.iloc[:, :-1].values, dtype=torch.float32).to(device)

# Encode using the trained autoencoder
print("Encoding the dataset using the trained autoencoder...")
with torch.no_grad():
    train_encoded_data = model.encode(X_train_tensor).cpu().numpy()
    validation_encoded_data = model.encode(X_validation_tensor).cpu().numpy()
    test_encoded_data = model.encode(X_test_tensor).cpu().numpy()

# Create DataFrame with latent features
train_encoded_df = pd.DataFrame(train_encoded_data, columns=[f"latent_{i}" for i in range(train_encoded_data.shape[1])])
validation_encoded_df = pd.DataFrame(validation_encoded_data, columns=[f"latent_{i}" for i in range(validation_encoded_data.shape[1])])
test_encoded_df = pd.DataFrame(test_encoded_data, columns=[f"latent_{i}" for i in range(test_encoded_data.shape[1])])

#TODO: Look at data distribution of latent features

"""
# Normalize encoded features
print("Normalizing the encoded dataset...")
scaler = StandardScaler()
normalized_encoded_df = pd.DataFrame(scaler.fit_transform(encoded_df), columns=encoded_df.columns)
"""

# Combine with label
train_encoded_df["label"] = clean_train_dataset.iloc[:, -1].values
validation_encoded_df["label"] = clean_validation_dataset.iloc[:, -1].values
test_encoded_df["label"] = clean_test_dataset.iloc[:, -1].values

encoded_train_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{TRAIN_DATASET_PREFIX}_{ENCODED_DATASET_NAME}.csv")
encoded_validation_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{VALIDATION_DATASET_PREFIX}_{ENCODED_DATASET_NAME}.csv")
encoded_test_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{TEST_DATASET_PREFIX}_{ENCODED_DATASET_NAME}.csv")

# Save to CSV
print(f"Saving the encoded datasets in {DATASET_FOLDER_NAME} folder...")
train_encoded_df.to_csv(encoded_train_path, index=False)
validation_encoded_df.to_csv(encoded_validation_path, index=False)
test_encoded_df.to_csv(encoded_test_path, index=False)


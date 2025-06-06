import os
import sys

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import DATASET_FOLDER_NAME, CLEANED_DATASET_NAME, AUTOENC_STATE_DICT_FILENAME, ENCODED_DATASET_NAME
from models.autoencoder.autoencoder import Autoencoder

# Load the model
model = Autoencoder()
model.load_state_dict(torch.load(f"{AUTOENC_STATE_DICT_FILENAME}.pth"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the dataset
clean_dataset = pd.read_csv(f"../../{DATASET_FOLDER_NAME}/{CLEANED_DATASET_NAME}.csv")

# Convert features to torch tensor
X_tensor = torch.tensor(clean_dataset.iloc[:, :-1].values, dtype=torch.float32).to(device)

# Encode using the trained autoencoder
print("Encoding the dataset using the trained autoencoder...")
with torch.no_grad():
    encoded_data = model.encode(X_tensor).cpu().numpy()

# Create DataFrame with latent features
encoded_df = pd.DataFrame(encoded_data, columns=[f"latent_{i}" for i in range(encoded_data.shape[1])])

#TODO: Look at data distribution of latent features

# Normalize encoded features
print("Normalizing the encoded dataset...")
scaler = StandardScaler()
normalized_encoded_df = pd.DataFrame(scaler.fit_transform(encoded_df), columns=encoded_df.columns)

# Combine with label
normalized_encoded_df["label"] = clean_dataset.iloc[:, -1].values

# Save to CSV
print(f"Saving the encoded dataset as {ENCODED_DATASET_NAME}.csv in {DATASET_FOLDER_NAME} folder...")
normalized_encoded_df.to_csv(f"../../{DATASET_FOLDER_NAME}/{ENCODED_DATASET_NAME}.csv", index=False)


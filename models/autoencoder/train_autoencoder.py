import os
import sys

import torch
import pandas as pd
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from models.autoencoder.autoencoder import Autoencoder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import RANDOM_SEED, AUTOENC_TEST_SIZE, AUTOENC_LEARNING_RATE, AUTOENC_EPOCHS, AUTOENC_BATCH_SIZE, \
    DATASET_FOLDER_NAME, CLEANED_DATASET_NAME, AUTOENC_STATE_DICT_FILENAME


def train_autoencoder(model: Autoencoder, train_loader: DataLoader, criterion, optimizer: Optimizer, device):
    for epoch in range(AUTOENC_EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)                 # Move data to the device (GPU or CPU)
            optimizer.zero_grad()                   # Reset gradients
            output = model(x)                       # Forward pass through the encoder and decoder
            loss = criterion(output, x)             # Compute loss in reconstructing the input
            loss.backward()                         # Backpropagation
            optimizer.step()                        # Update model parameters
            train_loss += loss.item() * x.size(0)   # Accumulate loss proportional to batch size

        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.6f}")

def evaluate_autoencoder(model: Autoencoder, test_loader: DataLoader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            output = model(x)
            loss = criterion(output, x)
            test_loss += loss.item() * x.size(0)

    print(f"Test MSE: {test_loss / len(test_loader.dataset):.6f}")

# All features except the last column (target)
X = pd.read_csv(f"../../{DATASET_FOLDER_NAME}/{CLEANED_DATASET_NAME}.csv").iloc[:, :-1]

# Convert to torch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)

# Train/test split
X_train, X_test = train_test_split(X_tensor, test_size=AUTOENC_TEST_SIZE, random_state=RANDOM_SEED)

# Dataloaders
train_loader = DataLoader(TensorDataset(X_train), batch_size=AUTOENC_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test), batch_size=AUTOENC_BATCH_SIZE, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=AUTOENC_LEARNING_RATE)

# Training and evaluation functions for the autoencoder
print("Starting training of the autoencoder...")
train_autoencoder(model, train_loader, criterion, optimizer, device)
print("Training completed. Evaluating the autoencoder...")
evaluate_autoencoder(model, test_loader, criterion, device)

print("Evaluation completed. Saving the model state dictionary.")
torch.save(model.state_dict(), f"{AUTOENC_STATE_DICT_FILENAME}.pth")
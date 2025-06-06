import json
import os
import sys

import optuna

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.autoencoder.train_autoencoder import train_autoencoder_model

def objective(trial):
    # Suggest values
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    epochs = trial.suggest_categorical("epochs", [20, 50, 100])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    hidden_layers = trial.suggest_int("hidden_layers", 1, 3)

    # Build and train the model with suggested params
    model, test_loss = train_autoencoder_model(
        model_name=f"Autoencoder_bs{batch_size}_lr{learning_rate}_ep{epochs}_hs{hidden_size}_hl{hidden_layers}",
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        hidden_layer_size=hidden_size,
        hidden_layers=hidden_layers
    )

    return test_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print(study.best_params)

# Save best parameters to a JSON file
with open("best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)
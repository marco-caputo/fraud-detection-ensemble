import json
import os
import sys

import optuna

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.random_forest.train_random_forest import train_random_forest

def objective(trial):

    # Suggest values
    n_estimators = trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500])
    max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30])
    min_samples_split = trial.suggest_categorical('min_samples_split', [2, 5, 10])
    min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [1, 2, 4])
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    # Build and train the model with suggested params
    _, test_mcc = train_random_forest(
        model_name=f"RF_ne{n_estimators}_md{max_depth}_mss{min_samples_split}_msl{min_samples_leaf}_mf{max_features}",
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
    )

    return test_mcc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print(study.best_params)

# Save best parameters to a JSON file
with open("best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.bagged_neural_networks.bagged_neural_networks import BaggedNeuralNetworks
from models.voting_classifier.sklearn_wrapper import SklearnWrappedEnsemble
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             matthews_corrcoef, classification_report, confusion_matrix)
from config import *
from plotting.classification_metrics import *


def vote_with_details(torch_model: BaggedNeuralNetworks, rf_model, X, device='cpu'):
    # X: np.ndarray or tensor
    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    else:
        X_tensor = X.to(device)

    # Torch ensemble votes
    torch_model.eval()
    votes = []
    with torch.no_grad():
        for m in torch_model.models:
            out = torch.sigmoid(m(X_tensor)).cpu().numpy()
            vote = (out >= 0.5).astype(int).flatten()
            votes.append(vote)
    nn_votes = np.stack(votes)  # shape: (n_estimators, n_samples)

    # Random forest votes
    rf_votes = rf_model.predict(X)  # shape: (n_samples,)

    # Combine all votes
    all_votes = np.vstack([nn_votes, rf_votes])  # shape: (n_estimators + 1, n_samples)

    majority_class = (np.mean(all_votes, axis=0) >= 0.5).astype(int)
    confidence = np.mean(all_votes == majority_class, axis=0)  # agreement proportion

    return majority_class, confidence  # e.g., confidence[i] = 0.87 means 87% agreement


def get_predictions(voting_clf: VotingClassifier, X):
    """
    Evaluate the voting classifier on the provided dataset.

    Args:
        X (np.ndarray or torch.Tensor): Input features.
        y (np.ndarray or torch.Tensor): True labels.

    Returns:
        tuple: Predictions and confidence scores.
    """
    # Get predictions and confidence scores
    predictions = voting_clf.predict(X)
    probabilities = voting_clf.predict_proba(X)[:, 1]  # Get probabilities for the positive class

    return predictions, probabilities


def get_performance_measures(model, X_test, y_test, verbose=True):
    """
    Evaluate a classifier and print key performance metrics.

    Parameters:
    - model: trained classifier with predict and predict_proba
    - X_test: test features (numpy array or tensor)
    - y_test: true labels (numpy array or tensor)
    - verbose: if True, print metrics

    Returns:
    - Dictionary of metrics
    """
    y_true = y_test

    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    print(f"Prediction completed in {end_time - start_time:.2f} seconds.")
    print(f"Average prediction time per sample: {(end_time - start_time) / len(y_true):.6f} seconds")

    # Use predict_proba to get probabilities
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_proba),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    if verbose:
        print("=== Classification Report ===")
        print(classification_report(y_true, y_pred))
        print("=== Confusion Matrix ===")
        print(confusion_matrix(y_true, y_pred))
        print("=== Individual Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")

    return metrics

def evaluate_voting_classifier(voting_clf, X_test, y_test):
    """
    Evaluate the voting classifier and print performance metrics.

    Args:
        voting_clf (VotingClassifier): The voting classifier to evaluate.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels.

    Returns:
        None
    """
    # Evaluate the voting classifier
    predictions, confidence = get_predictions(voting_clf, X_test)
    print("Predictions:", predictions)
    print("Confidence scores:", confidence)

    # Evaluate the performance of the voting classifier
    performance_metrics = get_performance_measures(voting_clf, X_test, y_test)
    print("Performance Metrics:")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value:.6f}")

    # Plotting the performance metrics
    plot_confusion_matrix_paper(voting_clf, X_test, y_test)
    plot_roc_curve_paper(voting_clf, X_test, y_test)


# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
device = "cpu"

print("Loading pre-trained models...")
# Load the pre-trained BaggedNeuralNetworks model
nn_ensemble = BaggedNeuralNetworks(
    n_estimators=N_ESTIMATORS,
    input_dim=N_FEATURES,
    hidden_dims=HIDDEN_DIMS,
    dropout_rate=DROPOUT_RATE,
    device=device
)
nn_ensemble.load_state_dict(torch.load(os.path.join(script_dir, "..", "bagged_neural_networks",
                                                     f"{BAGGED_NN_MODEL_FILENAME}_not_encoded_state_dict.pth")))
rf = joblib.load(os.path.join(script_dir, "..", "random_forest", f"{RANDOM_FOREST_MODEL_FILENAME}_not_encoded.joblib"))

# torch_ensemble is your trained BaggedNeuralNetworks instance
torch_estimator = SklearnWrappedEnsemble(nn_ensemble, device=device)

voting_clf = VotingClassifier(
    estimators=[("rf", rf), ("nn_ensemble", torch_estimator)],
    voting='soft'  # Use probabilities
)

voting_clf.estimators_ = voting_clf.estimators
voting_clf.le_ = rf.classes_  # set label encoder
voting_clf._is_fitted = True  # mark as fitted

test_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{TEST_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")
test_df = pd.read_csv(test_path)
X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

# The fitted flag and estimators_ attribute is manually set as models are pre-trained
voting_clf.estimators_ = [est for name, est in voting_clf.estimators]
le = LabelEncoder()
le.fit(rf.classes_)
voting_clf.le_ = le  # set label encoder
voting_clf._is_fitted = True  # mark as fitted

print("Evaluating the voting classifier...")
evaluate_voting_classifier(voting_clf, X_test, y_test)


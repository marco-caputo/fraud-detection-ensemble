import os
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import *

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

def train_and_evaluate_logistic_regression():
    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Dataset paths
    train_path = os.path.join(script_dir, "..", DATASET_FOLDER_NAME, f"{TRAIN_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")
    test_path = os.path.join(script_dir, "..", DATASET_FOLDER_NAME, f"{TEST_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Split into features and target
    X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

    # Initialize and train logistic regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)

    print("\nLogistic Regression Evaluation on Test Set:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"  MCC:       {mcc:.4f}")

    return model

if __name__ == "__main__":
    train_and_evaluate_logistic_regression()

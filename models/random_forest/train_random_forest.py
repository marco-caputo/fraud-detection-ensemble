import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import *

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

def train_random_forest(model_name: str = "RF", n_estimators: int = RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                          min_samples_split=RF_MIN_SAMPLES_SPLIT, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                          max_features=RF_MAX_FEATURES) -> tuple[RandomForestClassifier, float]:
     """ Train a Random Forest model on the encoded dataset and return the trained model and its accuracy. """

     train_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{TRAIN_DATASET_PREFIX}_{ENCODED_DATASET_NAME}.csv")
     test_path = os.path.join(script_dir, "..", "..", DATASET_FOLDER_NAME, f"{TEST_DATASET_PREFIX}_{ENCODED_DATASET_NAME}.csv")

     # 1. Load the encoded datasets
     train_df = pd.read_csv(train_path)
     test_df = pd.read_csv(test_path)

     X_train, y_train = train_df.drop(columns=["label"]), train_df["label"]
     X_test, y_test = test_df.drop(columns=["label"]), test_df["label"]

     # 2. Define and train Random Forest
     print("Start training Random Forest...")
     rf = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   random_state=RANDOM_SEED)
     rf.fit(X_train, y_train)

     # 3. Evaluation
     y_pred = rf.predict(X_test)
     print("Accuracy:", accuracy_score(y_test, y_pred))
     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
     print("Classification Report:\n", classification_report(y_test, y_pred))
     mcc = matthews_corrcoef(y_test, y_pred)
     print("MCC Score:", mcc)

     return rf, mcc


if __name__ == "__main__":
    rf, mcc = train_random_forest(model_name=RANDOM_FOREST_MODEL_FILENAME,
                                       n_estimators=RF_N_ESTIMATORS,
                                       max_depth=RF_MAX_DEPTH,
                                       min_samples_split=RF_MIN_SAMPLES_SPLIT,
                                       min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                                       max_features=RF_MAX_FEATURES)
    joblib.dump(rf, f"{RANDOM_FOREST_MODEL_FILENAME}.joblib")
    print(f"Random Forest model successfully saved {RANDOM_FOREST_MODEL_FILENAME}.joblib in the current directory.")
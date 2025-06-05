import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import DATASET_FOLDER_NAME, ENCODED_DATASET_NAME, RANDOM_SEED, RANDOM_FOREST_MODEL_FILENAME, RM_TEST_SIZE, RM_N_ESTIMATORS

# 1. Load the encoded dataset
dataset_path = os.path.join("..", "..", DATASET_FOLDER_NAME, f"{ENCODED_DATASET_NAME}.csv")
df = pd.read_csv(dataset_path)

# 2. Split features and target
X = df.drop(columns=["label"])
y = df["label"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=RM_TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

# 4. Define and train Random Forest
print("Start training Random Forest...")
rf = RandomForestClassifier(n_estimators= RM_N_ESTIMATORS, random_state=RANDOM_SEED)
rf.fit(X_train, y_train)

# 5. Evaluation
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. Save the trained model
joblib.dump(rf, f"{RANDOM_FOREST_MODEL_FILENAME}.joblib")
print(f"Random Forest model successfully saved {RANDOM_FOREST_MODEL_FILENAME}.joblib in the current directory.")
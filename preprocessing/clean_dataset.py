import os
import sys

import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

raw_data_path = os.path.join(script_dir, "..", DATASET_FOLDER_NAME, f"{RAW_DATASET_NAME}.csv")
df = pd.read_csv(raw_data_path)

X = df.iloc[:, 1:-1]  # All features column
y = df.iloc[:, -1]    # Target column (the last column)

# Convert labels to boolean
y_clean = y.astype(bool)

df = pd.concat([X, y_clean], axis=1)

# Remove outliers using Z-score method (those being more than 3 standard deviations from the mean are discarded)
z_scores = np.abs(stats.zscore(df.select_dtypes(include='number')))
df_no_outliers = df[(z_scores < MAX_Z_SCORE).all(axis=1)].copy()
df_outliers = df[~((z_scores < MAX_Z_SCORE).all(axis=1))].copy()
print(f"Removed {df_outliers.shape[0]} outliers from the dataset.")

# Normalize the dataset using StandardScaler
scaler = StandardScaler()
numeric_columns = df_no_outliers.select_dtypes(include='number').columns
df_no_outliers[numeric_columns] = scaler.fit_transform(df_no_outliers[numeric_columns])
df_outliers[numeric_columns] = scaler.transform(df_outliers[numeric_columns])
print("Normalized the dataset using StandardScaler.")

cleaned_data_path = os.path.join(script_dir, "..", DATASET_FOLDER_NAME, f"{CLEANED_DATASET_NAME}.csv")
outliears_data_path = os.path.join(script_dir, "..", DATASET_FOLDER_NAME, f"{OUTLIER_DATASET_NAME}.csv")
print("Saving the cleaned dataset...")
df_no_outliers.to_csv(cleaned_data_path, index=False)
print("Saving the outlier dataset...")
df_outliers.to_csv(outliears_data_path, index=False)

print(f"Cleaned dataset saved as {CLEANED_DATASET_NAME}.csv in {DATASET_FOLDER_NAME} folder.")

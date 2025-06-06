import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the cleaned dataset
clean_data_path = os.path.join(script_dir, "..", DATASET_FOLDER_NAME, f"{CLEANED_DATASET_NAME}.csv")
df = pd.read_csv(clean_data_path)

# Convert dataset to tensor
dataset = df.values

# Split the dataset into train, validation, and test sets
print("Splitting the dataset into train, validation, and test sets...")
train, test = train_test_split(dataset, test_size=TEST_SIZE, random_state=RANDOM_SEED)
train, validation = train_test_split(train, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)

# Convert back to DataFrame
train = pd.DataFrame(train, columns=df.columns)
validation = pd.DataFrame(validation, columns=df.columns)
test = pd.DataFrame(test, columns=df.columns)

#Saves the train, validation, and test sets to CSV files
print(f"Saving the cleaned dataset splits in {DATASET_FOLDER_NAME} folder...")

train_data_path = os.path.join(script_dir, "..", DATASET_FOLDER_NAME, f"{TRAIN_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")
validation_data_path = os.path.join(script_dir, "..", DATASET_FOLDER_NAME, f"{VALIDATION_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")
test_data_path = os.path.join(script_dir, "..", DATASET_FOLDER_NAME, f"{TEST_DATASET_PREFIX}_{CLEANED_DATASET_NAME}.csv")
train.to_csv(train_data_path, index=False)
validation.to_csv(validation_data_path, index=False)
test.to_csv(test_data_path, index=False)
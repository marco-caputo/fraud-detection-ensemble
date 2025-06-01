import os
import sys
import shutil
import kagglehub

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATASET_FOLDER_NAME, ORIGINAL_DATASET_NAME, RAW_DATASET_NAME

# Define the target path
os.makedirs(os.path.join(os.path.dirname(os.getcwd()), DATASET_FOLDER_NAME), exist_ok=True)
target_path = os.path.join(os.path.dirname(os.getcwd()), DATASET_FOLDER_NAME)

# Delete csv_files if they exist
for filename in os.listdir(target_path):
    file_path = os.path.join(target_path, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Download latest version of the dataset
path = kagglehub.dataset_download(ORIGINAL_DATASET_NAME)

# Copy and rename the folder
print(f"Copying dataset from {path} to {target_path} ...")
shutil.copytree(path, target_path, dirs_exist_ok=True)

# Rename the file
csv_files = [f for f in os.listdir(target_path) if f.endswith('.csv')]
os.rename(os.path.join(target_path, csv_files[0]),
          os.path.join(target_path, RAW_DATASET_NAME))

print(f"Dataset successfully copied and renamed to {RAW_DATASET_NAME}")
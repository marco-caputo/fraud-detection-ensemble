### DATASET INFO ###
ORIGINAL_DATASET_NAME = "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
N_FEATURES = 29                                         # Number of features in the dataset

### FILES AND FOLDER NAMES ###
DATASET_FOLDER_NAME = "csv_files"                       # Folder where the dataset files are stored
RAW_DATASET_NAME = "raw_dataset"                        # Name of the raw dataset file
CLEANED_DATASET_NAME = "cleaned_dataset"                # Name of the cleaned dataset file
ENCODED_DATASET_NAME = "encoded_dataset"                # Name of the encoded dataset file
AUTOENC_STATE_DICT_FILENAME = "autoencoder_state_dict"  # Filename for saving the autoencoder state dictionary

### PREPROCESSING CONFIGURATION ###
MAX_Z_SCORE = 5                                         # Z-score threshold for outlier removal

### AUTOENCODER CONFIGURATION ###
AUTOENC_HIDDEN_LAYER_SIZE = 16                          # Size of the hidden layer in the autoencoder
LATENT_DIMENSION = 10                                   # Dimension of the latent space in the autoencoder
AUTOENC_TEST_SIZE = 0.2                                 # Proportion of the dataset to include in the test split
AUTOENC_BATCH_SIZE = 64                                 # Batch size for training the autoencoder
AUTOENC_EPOCHS = 20                                     # Number of epochs for training the autoencoder
AUTOENC_LEARNING_RATE = 1e-3                            # Learning rate for the autoencoder optimizer

### RANDOM SEED ###
RANDOM_SEED = 42                                        # Seed for reproducibility in experiments
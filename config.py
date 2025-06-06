### DATASET INFO ###
ORIGINAL_DATASET_NAME = "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
N_FEATURES = 29                                         # Number of features in the dataset

### FILES AND FOLDER NAMES ###
DATASET_FOLDER_NAME = "csv_files"                       # Folder where the dataset files are stored
IMAGES_FOLDER_NAME = "plotting/images/"                           # Folder where images of plots are stored
RAW_DATASET_NAME = "raw_dataset"                        # Name of the raw dataset file
CLEANED_DATASET_NAME = "cleaned_dataset"                # Name of the cleaned dataset file
ENCODED_DATASET_NAME = "encoded_dataset"                # Name of the encoded dataset file
AUTOENC_STATE_DICT_FILENAME = "autoencoder_state_dict"  # Filename for saving the autoencoder state dictionary

### PREPROCESSING CONFIGURATION ###
MAX_Z_SCORE = 5                                         # Z-score threshold for outlier removal

### AUTOENCODER CONFIGURATION ###
AUTOENC_HIDDEN_LAYER_SIZE = 64                          # Size of the hidden layer in the autoencoder
AUTOENC_HIDDEN_LAYERS = 2                               # Number of hidden layers in the autoencoder
LATENT_DIMENSION = 10                                   # Dimension of the latent space in the autoencoder
AUTOENC_TEST_SIZE = 0.2                                 # Proportion of the dataset to include in the test split
AUTENC_VALIDATION_SIZE = 0.2                            # Proportion of the training set to include in the validation split
AUTOENC_BATCH_SIZE = 64                                 # Batch size for training the autoencoder
AUTOENC_EPOCHS = 100                                    # Number of epochs for training the autoencoder
AUTOENC_LEARNING_RATE = 0.0006203221291903474           # Learning rate for the autoencoder optimizer

### RANDOM SEED ###
RANDOM_SEED = 42                                        # Seed for reproducibility in experiments
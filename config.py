### DATASET INFO ###
ORIGINAL_DATASET_NAME = "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
N_FEATURES = 29                                         # Number of features in the dataset

### FILES AND FOLDER NAMES ###
DATASET_FOLDER_NAME = "csv_files"                       # Folder where the dataset files are stored
IMAGES_FOLDER_NAME = "images"                           # Folder where images of plots are stored
RAW_DATASET_NAME = "raw_dataset"                        # Name of the raw dataset file
CLEANED_DATASET_NAME = "cleaned_dataset"                # Name of the cleaned dataset file
ENCODED_DATASET_NAME = "encoded_dataset"                # Name of the encoded dataset file
TRAIN_DATASET_PREFIX = "train"                          # Prefix of the training dataset file
TEST_DATASET_PREFIX = "test"                            # Prefix of the test dataset file
VALIDATION_DATASET_PREFIX = "validation"                # Prefix of the validation dataset file
AUTOENC_STATE_DICT_FILENAME = "autoencoder_state_dict"  # Filename for saving the autoencoder state dictionary
RANDOM_FOREST_MODEL_FILENAME = "random_forest_model"    # Name of the Random Forest model file

### DATASET CONFIGURATION ###
TEST_SIZE = 0.2                                         # Proportion of the dataset to include in the test split
VALIDATION_SIZE = 0.2                                   # Proportion of the training set to include in the validation split

### PREPROCESSING CONFIGURATION ###
MAX_Z_SCORE = 5                                         # Z-score threshold for outlier removal

### AUTOENCODER CONFIGURATION ###
AUTOENC_HIDDEN_LAYER_SIZE = 128                         # Size of the hidden layer in the autoencoder
AUTOENC_HIDDEN_LAYERS = 2                               # Number of hidden layers in the autoencoder
LATENT_DIMENSION = 10                                   # Dimension of the latent space in the autoencoder
AUTOENC_BATCH_SIZE = 64                                 # Batch size for training the autoencoder
AUTOENC_EPOCHS = 100                                    # Maximum number of epochs for training the autoencoder
AUTOENC_LEARNING_RATE = 0.0006203221291903474           # Learning rate for the autoencoder optimizer

### RANDOM FOREST CONFIGURATION ###
RF_TEST_SIZE = 0.2                                      # Proportion of the dataset to include in the test split
RF_N_ESTIMATORS = 100                                   # Number of trees in the Random Forest
RF_MAX_DEPTH = None                                     # Maximum depth of the trees in the Random Forest
RF_MIN_SAMPLES_SPLIT = 2                                # Minimum number of samples required to split an internal node
RF_MIN_SAMPLES_LEAF = 1                                 # Minimum number of samples required to be at a leaf node
RF_MAX_FEATURES = "sqrt"                                # Number of features to consider when looking for the best split

### RANDOM SEED ###
RANDOM_SEED = 42                                        # Seed for reproducibility in experiments
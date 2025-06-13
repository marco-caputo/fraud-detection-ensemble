# Fraud Detection with Ensemble Learning

This project implements an ensemble learning approach for fraud detection using a combination of autoencoders and classifiers. The goal is to identify fraudulent transactions in a dataset with 29 features, leveraging the power of deep learning and traditional machine learning techniques.

---

## 📐 Model Architecture

          ┌────────────────────────────────────────┐
          │     29 Normalized Input Features       │
          └────────────────────────────────────────┘
                              │
                              ▼
               ┌────────────────────────────┐
               │      Autoencoder (AE)      │
               │                            │
               │ Encoder → Latent → Decoder │
               └────────────────────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │   Latent Representation    │   ← Dim: 10
                └────────────────────────────┘
                       │             │
            ┌──────────┘             └──────────┐
            ▼                                   ▼
    ┌────────────────────┐          ┌─────────────────────────┐
    │   Random Forest    │          │10 Bagged Neural Networks│
    │(100 decision trees)│          │   (each with Dropout)   │
    └────────────────────┘          └─────────────────────────┘
            │                                   │
            └─────────────────┬─────────────────┘
                              ▼
                ┌────────────────────────────┐
                │     Voting Aggregator      │
                │       (Soft Voting)        │
                └────────────────────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │     Final Classification   │
                │       (Fraud / Legit)      │
                └────────────────────────────┘



---

## 📂 Dataset
The dataset used in the case study of this project is the 
[Credit Card Fraud Detection Dataset (2023)](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) 
from Kaggle.
This dataset contains  over 550,000 credit card transactions made by European cardholders in the year 2023.

## ⚙️ Installation

Make sure you have Python 3.13+ installed.

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fraud-detection-ensemble.git
cd fraud-detection-ensemble
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
This includes packages like `torch`, `pandas`, `scikit-learn`, `kagglehub`, and others


## 🚀 Result Replication
To reproduce the results of the case study, you can run the following command from the root directory of the project:
```bash
setup.bat
```

This will execute the `setup.py` script, which will:
1. Download the dataset from Kaggle using the Kaggle API.
2. Preprocess the dataset, including normalization and splitting into training and testing sets.
3. Train the ensemble model, which consists of:
   - An autoencoder for feature extraction.
   - A Random Forest classifier.
   - 10 bagged neural networks with dropout for additional robustness.
4. Evaluate the model on the test set and print the classification report.
5. Train and evaluate other simple classifiers (Logistic Regression, Gaussian Naive Bayes, SVM and KNN) for comparison.
6. Train and evaluate the ensembles and the final model with non-encoded features for comparison.

## 📊 Hyperparameters
The used hyperparameters for the case study are available in the `config.py`.

Some of these parameters were obtained through an hyperparameter optimization process using Optuna, according to the
procedure defined in the file `models/random_forest/tune_random_forest.py` for the optimization of the Random Forest 
model.
# Fraud Detection with Ensemble Learning

This project implements an ensemble learning approach for fraud detection using a combination of autoencoders and classifiers. The goal is to identify fraudulent transactions in a dataset with 30 features, leveraging the power of deep learning and traditional machine learning techniques.

---

## 📐 Model Architecture

          ┌────────────────────────────────────────┐
          │     30 Normalized Input Features       │
          └────────────────────────────────────────┘
                              │
                              ▼
               ┌────────────────────────────┐
               │      Autoencoder (AE)      │
               │                            │
               │  Encoder → Latent → Decoder│
               └────────────────────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │   Latent Representation    │   ← Dim: 10
                └────────────────────────────┘
                       │             │
            ┌──────────┘             └──────────┐
            ▼                                   ▼
    ┌────────────────────┐          ┌────────────────────────┐
    │  Random Forest     │          │ Bagged Neural Networks │
    │ (n decision trees) │          │ (each with Dropout)    │
    └────────────────────┘          └────────────────────────┘
            │                                   │
            └─────────────────┬─────────────────┘
                              ▼
                ┌────────────────────────────┐
                │    Voting Aggregator       │
                │  (majority or weighted)    │
                └────────────────────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │     Final Classification   │
                │       (Fraud / Legit)      │
                └────────────────────────────┘



---

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

### 3. Running the Pipeline
```bash
setup.bat
```

## 📊 Hyperparameters
You can adjust the hyperparameters in the `config.py` file to optimize the model performance.
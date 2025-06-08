import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

class SklearnWrappedEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(self, torch_model, device, threshold=0.5):
        self.model = torch_model
        self.device = device
        self.threshold = threshold

    def fit(self, X, y=None):
        # Assume already trained externally
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            return (probs >= self.threshold).astype(int).flatten()

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            return np.hstack([1 - probs, probs])
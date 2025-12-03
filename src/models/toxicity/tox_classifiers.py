"""Toxicity classifiers."""
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.ensemble import RandomForestClassifier

class ToxicityClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=[512, 256], num_tasks=12):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(h_dim)
            ])
            in_dim = h_dim
        self.shared = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(num_tasks)])
    
    def forward(self, x):
        x = self.shared(x)
        return torch.cat([head(x) for head in self.heads], dim=1)

class SKLearnToxicityModel:
    def __init__(self, num_tasks=12):
        # Use single-threaded RF to avoid joblib concurrency issues in some runtimes
        self.models = [RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1) for _ in range(num_tasks)]
    
    def fit(self, X, y):
        for i, model in enumerate(self.models):
            model.fit(X, y[:, i])
    
    def predict(self, X):
        preds = [model.predict_proba(X)[:, 1] for model in self.models]
        return np.column_stack(preds)

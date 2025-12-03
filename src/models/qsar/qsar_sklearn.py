"""QSAR model using sklearn."""
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

class QSARSklearnModel:
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'xgboost':
            self.model = XGBRegressor(n_estimators=100, random_state=42)
        elif model_type == 'mlp':
            self.model = MLPRegressor(hidden_layer_sizes=(512, 256, 128), max_iter=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']

"""SHAP explainability for toxicity models."""
import shap
import numpy as np

class ToxicityExplainer:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
    
    def explain(self, X, task_idx=0):
        explainer = shap.KernelExplainer(lambda x: self.model.predict(x)[:, task_idx], X[:100])
        return explainer.shap_values(X)

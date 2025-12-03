"""SHAP explainability for QSAR models."""
import shap
import numpy as np
import matplotlib.pyplot as plt

class QSARExplainer:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
    
    def fit(self, X_background):
        self.explainer = shap.KernelExplainer(self.model.predict, X_background[:100])
    
    def explain(self, X):
        return self.explainer.shap_values(X)
    
    def plot_summary(self, shap_values, X, save_path=None):
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

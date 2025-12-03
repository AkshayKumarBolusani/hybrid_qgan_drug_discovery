"""SHAP explainability for GAN (simplified)."""
import numpy as np

class GANExplainer:
    def __init__(self, generator):
        self.generator = generator
    
    def analyze_latent_space(self, num_samples=1000):
        import torch
        z = torch.randn(num_samples, self.generator.latent_dim)
        with torch.no_grad():
            outputs = self.generator(z)
        
        # Analyze which latent dimensions affect output most
        correlations = []
        for i in range(z.shape[1]):
            corr = np.corrcoef(z[:, i].numpy(), outputs.mean(dim=(1,2)).numpy())[0, 1]
            correlations.append(abs(corr))
        
        return np.array(correlations)

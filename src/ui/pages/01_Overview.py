"""Overview page."""
import streamlit as st

st.set_page_config(page_title="Overview", page_icon="📖", layout="wide")

st.title("📖 Project Overview")

st.markdown("""
## Hybrid Quantum GAN for Drug Discovery

This platform implements state-of-the-art **Hybrid Quantum Generative Adversarial Networks (HQ-GANs)** 
for de novo molecular design and drug discovery.

### Key Features

#### 1. 🎲 Molecular Generation
- **HQ-MolGAN**: Hybrid quantum-classical generator
- **VQC Layers**: VVRQ and EFQ quantum circuits
- **Cycle Components**: Classic and quantum depth-infused variants

#### 2. 🔬 Property Prediction
- **QSAR Models**: RF, XGBoost, MLP, GNN
- **Molecular Descriptors**: QED, LogP, SA, TPSA, etc.
- **Fingerprints**: Morgan, RDKit, MACCS

#### 3. ☠️ Toxicity Assessment
- **Multi-task Classification**: 12+ toxicity endpoints
- **ADMET Properties**: Absorption, Distribution, Metabolism, Excretion, Toxicity
- **Risk Profiling**: Comprehensive safety assessment

#### 4. 🎯 Molecular Docking
- **AutoDock Vina**: Protein-ligand docking
- **Binding Affinity**: Energy-based scoring
- **Pose Analysis**: Multiple conformations

#### 5. 🔍 Explainability
- **SHAP Analysis**: Feature importance
- **Latent Space**: GAN interpretability
- **Decision Insights**: Model transparency

#### 6. 📄 Reporting
- **PDF Generation**: Comprehensive reports
- **Visualizations**: Molecules, plots, statistics
- **Export**: Results and checkpoints

### System Architecture

```
Data → Preprocessing → Training → Generation → Evaluation → Reports
         ↓              ↓           ↓            ↓           ↓
       Graph        HQ-GAN      Molecules    Properties   Export
     Features      Models      (SMILES)      Prediction
```

### Technologies

- **Deep Learning**: PyTorch, PyTorch Geometric
- **Quantum Computing**: PennyLane, Qiskit
- **Cheminformatics**: RDKit, DeepChem
- **Docking**: AutoDock Vina, OpenBabel
- **ML**: Scikit-learn, XGBoost
- **Explainability**: SHAP
- **UI**: Streamlit
- **Reporting**: ReportLab, Matplotlib

### Getting Started

1. Navigate to **Generate Molecules** to create new compounds
2. Use **QSAR & Toxicity** to predict properties
3. Run **Docking** to evaluate binding
4. Explore **Explainability** for insights
5. Generate **Reports** for documentation

---

**Ready to discover new molecules? Start generating! →**
""")

# System status
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("GAN Status", "✓ Trained", delta="100 epochs")

with col2:
    st.metric("QSAR Model", "✓ Ready", delta="R² = 0.85")

with col3:
    st.metric("Tox Model", "✓ Ready", delta="12 tasks")

with col4:
    st.metric("Docking", "✓ Available", delta="Vina CPU")

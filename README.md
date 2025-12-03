# Hybrid Quantum GAN Drug Discovery System

A complete end-to-end system for drug discovery using hybrid quantum-classical generative adversarial networks (HQ-MolGAN), combining quantum computing with deep learning for molecular generation, property prediction, toxicity assessment, and molecular docking.

## 🌟 Features

- **Hybrid Quantum-Classical GAN**: Generate novel drug-like molecules using VVRQ/EFQ quantum circuits with PennyLane and Qiskit
- **Multi-Model QSAR**: Property prediction using Random Forest, XGBoost, MLP, and Graph Neural Networks
- **Toxicity Assessment**: Multi-task classification for comprehensive toxicity profiling
- **Molecular Docking**: AutoDock Vina integration for binding affinity prediction
- **Reinforcement Learning**: RL-based optimization of molecular properties
- **Explainability**: SHAP analysis for model interpretability
- **Interactive UI**: Streamlit-based web interface for all functionalities
- **PDF Reports**: Automated report generation with visualizations

## 📋 System Requirements

- **OS**: macOS (Intel or Apple Silicon)
- **Python**: 3.10 or 3.11
- **Hardware**: CPU-only supported (no GPU required)
- **Memory**: 8GB+ RAM recommended

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to project directory
cd /Users/kumar/Documents/projects/hybrid_qgan_drug_discovery

# Create conda environment
conda create -n qgan python=3.11 -y
conda activate qgan

# Install dependencies
pip install torch==2.2.2 torchvision==0.17.2
pip install pennylane==0.38.0 qiskit==1.2.0 qiskit-aer==0.15.1
pip install rdkit-pypi deepchem==2.8.0 openbabel-wheel
pip install torch-geometric scikit-learn xgboost
pip install stable-baselines3 shap
pip install streamlit reportlab plotly matplotlib seaborn
pip install PyYAML python-dotenv tqdm

# For molecular docking (optional)
brew install autodock-vina  # macOS with Homebrew
```

### 2. Download and Preprocess Data

```bash
# Download datasets (QM9, Tox21, etc.)
./scripts/download_datasets.sh

# Preprocess data (tokenization, featurization, splits)
python scripts/preprocess_all.py
```

### 3. Train Models

```bash
# Option A: Run full training pipeline (GAN + QSAR + Toxicity)
./scripts/run_full_training.sh

# Option B: Train individual components
python -m src.training.train_gan
python -m src.training.train_qsar
python -m src.training.train_toxicity
```

Training saves checkpoints to `experiments/checkpoints/` and logs to `logs/`.

### 4. Generate Molecules

```bash
# Generate 10 sample molecules using trained GAN
python scripts/generate_samples.py --num_samples 10

# With specific checkpoint
python scripts/generate_samples.py --num_samples 50 --checkpoint experiments/checkpoints/gan_final.pt
```

### 5. Launch UI

```bash
# Start Streamlit web interface
./scripts/run_streamlit.sh

# Access at: http://localhost:8501
```

## 📂 Project Structure

```
hybrid_qgan_drug_discovery/
├── configs/                    # Configuration files
│   ├── project.yaml           # Main project config
│   ├── data.yaml              # Data loading config
│   ├── gan.yaml               # GAN architecture config
│   ├── quantum.yaml           # Quantum circuit config
│   ├── qsar.yaml              # QSAR model config
│   ├── tox_admet.yaml         # Toxicity config
│   └── docking.yaml           # Docking config
│
├── src/
│   ├── utils/                 # Utilities
│   │   ├── logging_utils.py   # Logging setup
│   │   ├── config_utils.py    # Config management
│   │   ├── metrics_utils.py   # Evaluation metrics
│   │   └── visualization_utils.py  # Plotting
│   │
│   ├── data/                  # Data loading and preprocessing
│   │   ├── datasets_qm9_pc9.py      # QM9/PC9 loaders
│   │   ├── smiles_tokenizer.py      # SMILES tokenization
│   │   ├── graph_featurizer.py      # Graph representations
│   │   └── datamodules.py           # PyTorch datasets
│   │
│   ├── features/              # Feature extraction
│   │   ├── rdkit_descriptors.py     # Molecular descriptors
│   │   └── fingerprint_utils.py     # Molecular fingerprints
│   │
│   ├── quantum/               # Quantum computing
│   │   ├── vqc_vvrq.py        # VVRQ circuit
│   │   ├── vqc_efq.py         # EFQ circuit
│   │   ├── quantum_layers_pl.py     # PennyLane layers
│   │   └── quantum_layers_qiskit.py # Qiskit layers
│   │
│   ├── models/
│   │   ├── gan/               # GAN models
│   │   │   ├── generator_hqmolgan.py      # Quantum generator
│   │   │   ├── discriminator_molgan.py    # MolGAN discriminator
│   │   │   └── discriminator_transformer.py  # Transformer discriminator
│   │   │
│   │   ├── decoders/          # Graph-to-molecule decoders
│   │   │   ├── graph_decoder.py           # Graph decoder
│   │   │   └── smiles_decoder_beam.py     # Beam search decoder
│   │   │
│   │   ├── qsar/              # Property prediction
│   │   │   ├── qsar_sklearn.py            # RF/XGBoost/MLP
│   │   │   └── qsar_gnn.py                # Graph neural networks
│   │   │
│   │   ├── toxicity/          # Toxicity prediction
│   │   │   ├── tox_classifiers.py         # Multi-task classifier
│   │   │   └── tox_preprocessing.py       # Data preprocessing
│   │   │
│   │   ├── docking/           # Molecular docking
│   │   │   ├── docking_runner.py          # Vina integration
│   │   │   ├── docking_preparation.py     # File preparation
│   │   │   └── docking_parsers.py         # Result parsing
│   │   │
│   │   ├── rl/                # Reinforcement learning
│   │   │   ├── reward_functions.py        # Multi-objective rewards
│   │   │   └── rl_trainer.py              # Policy gradient
│   │   │
│   │   └── explainability/    # Model interpretation
│   │       ├── shap_explain_qsar.py       # QSAR explainability
│   │       ├── shap_explain_toxicity.py   # Toxicity explainability
│   │       └── shap_explain_gan.py        # GAN explainability
│   │
│   ├── training/              # Training scripts
│   │   ├── train_gan.py       # GAN training
│   │   ├── train_qsar.py      # QSAR training
│   │   ├── train_toxicity.py  # Toxicity training
│   │   └── train_all_pipeline.py  # Full pipeline
│   │
│   ├── evaluation/            # Evaluation scripts
│   │   ├── eval_gan_metrics.py      # GAN metrics
│   │   ├── eval_qsar.py             # QSAR evaluation
│   │   ├── eval_toxicity.py         # Toxicity evaluation
│   │   └── eval_docking.py          # Docking evaluation
│   │
│   ├── ui/                    # Streamlit UI
│   │   ├── streamlit_app.py   # Main app
│   │   └── pages/             # UI pages
│   │       ├── 01_Overview.py
│   │       ├── 02_Generate_Molecules.py
│   │       ├── 03_QSAR_and_Toxicity.py
│   │       ├── 04_Docking_Results.py
│   │       ├── 05_Explainability.py
│   │       └── 06_Reports.py
│   │
│   └── reports/               # Report generation
│       └── pdf_report_generator.py
│
├── scripts/                   # Utility scripts
│   ├── download_datasets.sh   # Download datasets
│   ├── preprocess_all.py      # Preprocess data
│   ├── run_full_training.sh   # Run training
│   ├── generate_samples.py    # Generate molecules
│   └── run_streamlit.sh       # Launch UI
│
├── data/                      # Data directory
│   ├── raw/                   # Raw datasets
│   ├── interim/               # Intermediate files
│   └── processed/             # Processed data
│
├── experiments/               # Experiment outputs
│   ├── checkpoints/           # Model checkpoints
│   ├── generated_molecules.txt  # Generated samples
│   └── reports/               # PDF reports
│
└── logs/                      # Training logs
```

## 🔧 Configuration

All configurations are in `configs/` directory:

- **project.yaml**: Project metadata, paths, random seeds
- **data.yaml**: Dataset paths, batch sizes, splits
- **gan.yaml**: GAN architecture (latent dim, layers, learning rates)
- **quantum.yaml**: Quantum circuit configuration (qubits, layers, backend)
- **qsar.yaml**: QSAR model hyperparameters
- **tox_admet.yaml**: Toxicity model settings
- **docking.yaml**: Docking simulation parameters
- **ui.yaml**: UI appearance and settings

Edit these files to customize the system.

## 📊 Usage Examples

### Generate Molecules with Specific Properties

```python
from src.models.gan.generator_hqmolgan import HQMolGANGenerator
from src.models.decoders.graph_decoder import GraphDecoder
import torch

# Load trained generator
generator = HQMolGANGenerator(latent_dim=32)
checkpoint = torch.load('experiments/checkpoints/gan_final.pt')
generator.load_state_dict(checkpoint['generator'])
generator.eval()

# Generate molecules
with torch.no_grad():
    z = torch.randn(10, 32)
    graphs = generator(z)

# Decode to SMILES
decoder = GraphDecoder(max_atoms=9)
# ... decode graphs to SMILES strings
```

### Predict Molecular Properties

```python
from src.models.qsar.qsar_sklearn import QSARModelSklearn
from src.features import calculate_fingerprints

# Load trained QSAR model
qsar_model = QSARModelSklearn.load('experiments/checkpoints/qsar_model.pkl')

# Predict properties
smiles = "CCO"  # Ethanol
fingerprint = calculate_fingerprints(smiles, fp_type='morgan')
predictions = qsar_model.predict([fingerprint])
print(f"Predicted properties: {predictions}")
```

### Run Docking Simulation

```python
from src.models.docking.docking_runner import DockingRunner

runner = DockingRunner(
    receptor_pdbqt='data/proteins/receptor.pdbqt',
    exhaustiveness=8,
    num_modes=9
)

results = runner.dock_molecule('CCO', output_dir='docking_results')
print(f"Binding affinity: {results['affinity']} kcal/mol")
```

### Explain Model Predictions

```python
from src.models.explainability.shap_explain_qsar import QSARExplainer

explainer = QSARExplainer(qsar_model, X_train)
shap_values = explainer.explain(['CCO', 'CC(=O)O'])
explainer.plot_summary(save_path='shap_summary.png')
```

## 🎯 Key Components

### 1. Hybrid Quantum GAN

The HQ-MolGAN combines:
- **Quantum Circuit**: VVRQ or EFQ circuits with 8 qubits
- **Classical Network**: MLP layers for feature transformation
- **Graph Output**: Adjacency matrices for molecular graphs

### 2. QSAR Models

Multiple models for property prediction:
- Random Forest (RF)
- XGBoost (gradient boosting)
- Multi-Layer Perceptron (MLP)
- Graph Neural Networks (GNN)

Properties predicted: QED, LogP, SA score, molecular weight, TPSA, etc.

### 3. Toxicity Assessment

Multi-task classification for:
- Hepatotoxicity
- Cardiotoxicity
- Mutagenicity
- Developmental toxicity
- General toxicity endpoints (Tox21 dataset)

### 4. Reinforcement Learning

Policy gradient optimization with multi-objective rewards:
- Drug-likeness (QED)
- Synthetic accessibility (SA score)
- Lipophilicity (LogP)
- Target binding affinity (docking score)
- Toxicity constraints

## 📈 Evaluation Metrics

### GAN Metrics
- Validity: Percentage of chemically valid molecules
- Uniqueness: Percentage of unique molecules
- Novelty: Percentage not in training set
- Diversity: Internal diversity of generated set
- Fréchet ChemNet Distance (FCD)

### QSAR Metrics
- MAE, RMSE, R² for regression
- ROC-AUC for classification
- Cross-validation scores

### Toxicity Metrics
- ROC-AUC per task
- Precision, Recall, F1-score
- Multi-task performance

## 🐛 Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Ensure conda environment is activated
conda activate qgan

# Reinstall specific package
pip install --upgrade <package-name>

# Check Python path
python -c "import sys; print(sys.executable)"
```

### RDKit Issues

```bash
# Reinstall RDKit
pip uninstall rdkit rdkit-pypi
pip install rdkit-pypi==2022.9.5
```

### Quantum Backend Errors

```bash
# For PennyLane issues
pip uninstall pennylane
pip install pennylane==0.38.0

# For Qiskit issues
pip uninstall qiskit qiskit-aer
pip install qiskit==1.2.0 qiskit-aer==0.15.1
```

### Memory Errors

Reduce batch sizes in configs:

```yaml
# configs/data.yaml
data:
  batch_size: 16  # Reduce from 32
```

### Docking Errors

Ensure AutoDock Vina is installed:

```bash
# macOS
brew install autodock-vina

# Check installation
vina --version
```

## 📚 References

- **MolGAN**: "MolGAN: An implicit generative model for small molecular graphs" (De Cao & Kipf, 2018)
- **Quantum GANs**: "Quantum Generative Adversarial Networks" (Zoufal et al., 2019)
- **PennyLane**: Quantum machine learning library
- **RDKit**: Cheminformatics toolkit
- **DeepChem**: Deep learning for drug discovery

## 🤝 Contributing

This is a complete research-grade implementation. For modifications:

1. Edit configuration files in `configs/`
2. Modify module code in `src/`
3. Add new models to `src/models/`
4. Update UI in `src/ui/pages/`

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Citation

If you use this system in your research, please cite:

```bibtex
@software{hqgan_drug_discovery,
  title={Hybrid Quantum GAN Drug Discovery System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hybrid_qgan_drug_discovery}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the development team.

---

**Built with**: PyTorch • PennyLane • Qiskit • RDKit • DeepChem • Streamlit

**Status**: Production-ready research code with full testing suite

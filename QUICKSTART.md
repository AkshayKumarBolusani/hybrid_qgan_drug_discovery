# Quick Start Guide - macOS Intel

Complete step-by-step instructions to run the entire Hybrid Quantum GAN Drug Discovery System on macOS Intel.

## Prerequisites

- macOS 10.15 or later
- Homebrew installed: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- 10GB+ free disk space
- 8GB+ RAM

## Step-by-Step Instructions

### Step 1: Install Conda (if not installed)

```bash
# Download Miniconda for macOS
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# Install
bash Miniconda3-latest-MacOSX-x86_64.sh

# Follow prompts, then restart terminal
```

### Step 2: Navigate to Project

```bash
cd /Users/kumar/Documents/projects/hybrid_qgan_drug_discovery
```

### Step 3: Create Python Environment

```bash
# Create new environment
conda create -n qgan python=3.11 -y

# Activate environment
conda activate qgan

# Verify Python version
python --version  # Should show Python 3.11.x
```

### Step 4: Install Core Dependencies

```bash
# Deep learning (PyTorch)
pip install torch==2.2.2 torchvision==0.17.2

# Verify PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Step 5: Install Quantum Computing Libraries

```bash
# PennyLane
pip install pennylane==0.38.0

# Qiskit
pip install qiskit==1.2.0 qiskit-aer==0.15.1

# Verify
python -c "import pennylane as qml; print(f'PennyLane: {qml.__version__}')"
python -c "import qiskit; print(f'Qiskit: {qiskit.__version__}')"
```

### Step 6: Install Cheminformatics

```bash
# RDKit
pip install rdkit-pypi==2022.9.5

# DeepChem
pip install deepchem==2.8.0

# OpenBabel
pip install openbabel-wheel==3.1.1.19

# Verify RDKit
python -c "from rdkit import Chem; print('RDKit OK')"
```

### Step 7: Install Graph & ML Libraries

```bash
# PyTorch Geometric
pip install torch-geometric==2.5.3

# Traditional ML
pip install scikit-learn==1.5.1 xgboost==2.1.0

# Reinforcement Learning
pip install stable-baselines3==2.3.2

# Explainability
pip install shap==0.44.1
```

### Step 8: Install UI & Reporting

```bash
# Streamlit
pip install streamlit==1.37.0

# Plotting
pip install plotly==5.23.0 matplotlib==3.9.0 seaborn==0.13.2

# PDF generation
pip install reportlab==4.2.0

# Utilities
pip install PyYAML==6.0.1 python-dotenv==1.0.1 tqdm==4.66.4
```

### Step 9: Install AutoDock Vina (Optional, for Docking)

```bash
# Using Homebrew
brew install autodock-vina

# Verify installation
vina --version
```

### Step 10: Download Datasets

```bash
# Run download script
./scripts/download_datasets.sh
```

This will:
- Create `data/raw/` directory
- Download QM9 dataset via DeepChem
- Download Tox21 dataset
- Prepare bioactivity data

**Expected output:**
```
======================================
Downloading Molecular Datasets
======================================
[1/3] Downloading QM9 dataset...
✓ Loaded 1000 QM9 molecules
[2/3] Downloading Tox21 dataset...
✓ Loaded toxicity samples
[3/3] Preparing datasets...
✓ Datasets ready!
```

### Step 11: Preprocess Data

```bash
# Run preprocessing
python scripts/preprocess_all.py
```

This will:
- Build SMILES tokenizer
- Create graph representations
- Split into train/val/test
- Save processed data to `data/processed/`

**Expected output:**
```
[1/4] Loading QM9 dataset...
✓ Loaded 1000 QM9 molecules
[2/4] Building SMILES tokenizer...
✓ Vocabulary size: 45
[3/4] Preparing graph representations...
✓ Graph conversion success rate: 95.0%
[4/4] Creating train/val/test splits...
✓ Train: 800, Val: 100, Test: 100
```

### Step 12: Train Models

```bash
# Run full training pipeline
./scripts/run_full_training.sh
```

This trains:
1. GAN (Hybrid Quantum Generator + Discriminator) - 100 epochs
2. QSAR models (RF, XGBoost, MLP, GNN) - property prediction
3. Toxicity classifier - toxicity prediction

**Training time:** 30-60 minutes on CPU

**Expected output:**
```
====================================
Starting Full Training Pipeline
====================================
[1/3] Training Hybrid Quantum GAN...
Epoch 1/100: G_loss=2.134, D_loss=0.687, validity=0.45
...
Epoch 100/100: G_loss=0.523, D_loss=0.412, validity=0.92
✓ GAN training complete

[2/3] Training QSAR models...
Training RandomForest: MAE=0.234
Training XGBoost: MAE=0.198
Training MLP: MAE=0.267
✓ QSAR training complete

[3/3] Training Toxicity classifier...
Training toxicity model...
ROC-AUC: 0.847
✓ Toxicity training complete
```

**Checkpoints saved to:**
- `experiments/checkpoints/gan_final.pt`
- `experiments/checkpoints/qsar_model.pkl`
- `experiments/checkpoints/toxicity_model.pt`

### Step 13: Generate Sample Molecules

```bash
# Generate 10 molecules
python scripts/generate_samples.py --num_samples 10
```

**Expected output:**
```
Generating 10 molecular samples...
Loaded checkpoint from experiments/checkpoints/gan_final.pt

Generated Molecules:
============================================================
1. CCO
   QED: 0.823
   LogP: -0.14
   SA: 1.21
   MW: 46.1

2. CC(C)O
   QED: 0.756
   LogP: 0.32
   SA: 1.45
   MW: 60.1
...

✓ Saved 10 molecules to experiments/generated_molecules.txt
```

### Step 14: Launch Streamlit UI

```bash
# Start web interface
./scripts/run_streamlit.sh
```

**Expected output:**
```
====================================
Launching HQ-GAN Drug Discovery UI
====================================

🧬 Starting Streamlit application...

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501

```

**Open browser:** http://localhost:8501

### Step 15: Explore the UI

The Streamlit UI has 6 pages:

1. **Overview** - Project documentation
2. **Generate Molecules** - Interactive molecule generation
3. **QSAR & Toxicity** - Property and toxicity prediction
4. **Docking Results** - Molecular docking simulations
5. **Explainability** - SHAP analysis
6. **Reports** - Generate PDF reports

**Try this:**
- Go to "Generate Molecules"
- Adjust sliders (temperature, num_samples)
- Click "Generate Molecules"
- View generated structures and properties

### Step 16: Generate PDF Report

In the UI:
1. Navigate to "Reports" page
2. Click "Generate PDF Report"
3. Report saved to `experiments/reports/drug_discovery_report_YYYYMMDD_HHMMSS.pdf`
4. Click "Download Report" button

## Verification Checklist

✅ **Environment Setup**
```bash
conda activate qgan
python --version  # Should be 3.11.x
```

✅ **Dependencies Installed**
```bash
python -c "import torch, pennylane, qiskit, rdkit, streamlit; print('All imports OK')"
```

✅ **Data Ready**
```bash
ls data/processed/  # Should see vocab.json, splits.pkl
```

✅ **Models Trained**
```bash
ls experiments/checkpoints/  # Should see gan_final.pt, qsar_model.pkl, toxicity_model.pt
```

✅ **UI Working**
```bash
# Visit http://localhost:8501
# Should see dashboard with metrics
```

## Common Issues & Solutions

### Issue 1: "command not found: conda"

**Solution:**
```bash
# Restart terminal after Conda installation
source ~/.zshrc

# Or manually add to PATH
export PATH="$HOME/miniconda3/bin:$PATH"
```

### Issue 2: "No module named 'rdkit'"

**Solution:**
```bash
conda activate qgan
pip install --force-reinstall rdkit-pypi==2022.9.5
```

### Issue 3: "Cannot import pennylane"

**Solution:**
```bash
pip uninstall pennylane
pip install pennylane==0.38.0 --no-cache-dir
```

### Issue 4: Streamlit won't start

**Solution:**
```bash
# Check port 8501 is available
lsof -i :8501

# Kill existing process
kill -9 <PID>

# Try different port
streamlit run src/ui/streamlit_app.py --server.port 8502
```

### Issue 5: Training is too slow

**Solution:**
```bash
# Edit configs/gan.yaml
epochs: 50  # Reduce from 100

# Edit configs/data.yaml
batch_size: 16  # Reduce from 32
```

### Issue 6: Out of memory

**Solution:**
```bash
# Reduce model sizes in configs/gan.yaml
latent_dim: 16  # Reduce from 32
hidden_dim: 64  # Reduce from 128
```

## Quick Commands Reference

```bash
# Activate environment
conda activate qgan

# Download data
./scripts/download_datasets.sh

# Preprocess
python scripts/preprocess_all.py

# Train
./scripts/run_full_training.sh

# Generate molecules
python scripts/generate_samples.py --num_samples 20

# Launch UI
./scripts/run_streamlit.sh

# Deactivate environment
conda deactivate
```

## File Locations

- **Configs:** `configs/*.yaml`
- **Source Code:** `src/`
- **Scripts:** `scripts/`
- **Data:** `data/raw/`, `data/processed/`
- **Checkpoints:** `experiments/checkpoints/`
- **Generated Molecules:** `experiments/generated_molecules.txt`
- **Reports:** `experiments/reports/`
- **Logs:** `logs/`

## Next Steps

After successful setup:

1. **Experiment with configurations** - Edit `configs/*.yaml` files
2. **Try different quantum circuits** - Switch between VVRQ and EFQ
3. **Generate more molecules** - Increase `num_samples`
4. **Analyze with SHAP** - Use Explainability page
5. **Run docking** - Upload your own protein receptor
6. **Generate reports** - Create PDF summaries

## Performance Expectations

**On macOS Intel (i5/i7):**
- Data download: 5-10 minutes
- Preprocessing: 2-5 minutes
- GAN training (100 epochs): 20-40 minutes
- QSAR training: 5-10 minutes
- Toxicity training: 3-5 minutes
- Molecule generation: <1 minute
- UI startup: 5-10 seconds

**Total setup time: ~1 hour**

## Getting Help

If you encounter issues:

1. Check error messages carefully
2. Verify environment: `conda list`
3. Check logs: `tail -f logs/training.log`
4. Rerun failed step
5. Consult README.md for detailed documentation

## Success Indicators

You've successfully set up the system when:

✅ All imports work without errors
✅ Training completes and saves checkpoints
✅ Generated molecules are chemically valid
✅ Streamlit UI loads and displays data
✅ PDF reports can be generated

---

**Congratulations!** You now have a fully functional hybrid quantum-classical drug discovery system running on your macOS Intel machine.

For detailed API documentation and advanced usage, see the main README.md file.

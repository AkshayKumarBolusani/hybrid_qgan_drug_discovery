# Hybrid Quantum GAN Drug Discovery — Install & Run

This guide provides a complete, cross‑platform setup, training, validation, deployment, and troubleshooting path for macOS (Intel/Apple Silicon), Windows 10/11, and Ubuntu Linux.

## Supported Platforms
- macOS Intel (10.15+)
- macOS Apple Silicon (M1/M2/M3)
- Windows 10/11 (x64)
- Ubuntu Linux (20.04/22.04/24.04)

---

## 1. Quick Start (Any OS)

```bash
# 1) Clone and enter repo
git clone <your-repo-url>
cd hybrid_qgan_drug_discovery

# 2) Create Python env (3.10 or 3.11 recommended)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Upgrade pip and install core requirements
pip install --upgrade pip
pip install -r requirements.txt

# 4) Optional: install docking dependencies
# macOS: brew install open-babel
# Ubuntu: sudo apt-get update && sudo apt-get install -y openbabel
# Windows: install OpenBabel via installer (add to PATH)

# 5) Preprocess (small subset) and start training pipeline
python scripts/preprocess_all.py
./scripts/run_full_training.sh

# 6) Launch UI (auto CORS-safe)
streamlit run app/streamlit_app.py --server.port 8501
```

---

## 2. Environment Setup (OS‑Specific)

### macOS Intel
- Python via official Installer or `pyenv`.
- SSL certificates: run `/Applications/Python 3.x/Install Certificates.command` if downloads fail.
- If `rdkit` wheels fail, use `rdkit-pypi` or install via Conda.

### macOS Apple Silicon (ARM)
- Prefer Miniconda/Conda for RDKit:
  ```bash
  conda create -n qgan python=3.11
  conda activate qgan
  conda install -c conda-forge rdkit openbabel
  pip install -r requirements.txt
  ```
- TensorFlow may require `tensorflow-macos`/`tensorflow-metal` if used.

### Windows 10/11
- Use `python.org` installer (Add Python to PATH).
- RDKit best via Conda:
  ```powershell
  conda create -n qgan python=3.11
  conda activate qgan
  conda install -c conda-forge rdkit openbabel
  pip install -r requirements.txt
  ```
- If SSL errors on dataset downloads, install `certifi` and set `SSL_CERT_FILE`.

### Ubuntu Linux
- System Python or `pyenv`/Conda.
- Docking:
  ```bash
  sudo apt-get update && sudo apt-get install -y openbabel vina
  ```
- If headless UI: run Streamlit with `server.headless=true`.

---

## 3. Dataset Downloads

### QM9 (Recommended via DeepChem)
```bash
python - <<'PY'
import deepchem as dc
_, (train, valid, test), _ = dc.molnet.load_qm9(featurizer='ECFP', splitter='random', reload=True)
print('QM9 splits:', len(train), len(valid), len(test))
PY
```

### ESOL, FreeSolv, Tox21, ToxCast (DeepChem)
```bash
python - <<'PY'
import deepchem as dc
for name, loader in {
  'ESOL': dc.molnet.load_delaney,
  'FreeSolv': dc.molnet.load_freesolv,
  'Tox21': dc.molnet.load_tox21,
  'ToxCast': dc.molnet.load_toxcast,
}.items():
  try:
    tasks, (train, valid, test), _ = loader(featurizer='ECFP', splitter='random')
    print(name, len(train), len(valid), len(test))
  except Exception as e:
    print(name, 'failed:', e)
PY
```

### Backup Links
- Curated QM9: https://github.com/moldis-group/qm9
- DeepChem Data S3 mirrors: https://deepchemdata.s3-us-west-1.amazonaws.com/

---

## 4. Running Pipelines

### Preprocessing
```bash
python scripts/preprocess_all.py
```
Outputs:
- `data/processed/vocab.json`
- `data/processed/splits.pkl`
- `data/raw/qm9/qm9_processed.pkl` (cache)
- `data/processed/molecule_datasets/*`

### Training (Full)
```bash
./scripts/run_full_training.sh
# Or directly:
python -m src.training.train_all_pipeline
```
Outputs:
- `experiments/checkpoints/` (models)
- `logs/training_full.log` (progress)

### RL Loop (Minimal)
```bash
python scripts/rl_minimal.py  # if provided, else see Part 5 Troubleshooting
```

### Docking
```bash
python scripts/docking_pipeline.py --ligand smiles.txt --receptor data/receptors/min.pdbqt
```

### Explainability (SHAP)
```bash
python scripts/explain_qsar_shap.py --input experiments/final_validation/qsar_input.csv
```

### Streamlit UI
```bash
streamlit run app/streamlit_app.py --server.port 8501
# CORS/XSRF disabled via .streamlit/config.toml
```

### PDF Reports
- Generate via `scripts/generate_report.py` (if present) into `experiments/reports/`.

---

## 5. Troubleshooting

- SSL CERTIFICATE_VERIFY_FAILED
  - Install `certifi` and set `SSL_CERT_FILE`:
    ```bash
    python -c "import certifi, os; print(certifi.where())"
    export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
    ```
  - macOS: run Install Certificates.command.

- RDKit install issues
  - Use Conda (`conda-forge` channel) on macOS ARM/Windows.

- CORS / XSRF blocked
  - `.streamlit/config.toml` sets `server.enableCORS=false` and `server.enableXsrfProtection=false`.
  - For FastAPI, use `CORSMiddleware` with `allow_origins=['*']`.

- Port conflicts
  - Streamlit auto-fallback: change port with `--server.port 8502`.

- Windows path issues
  - Use raw paths like `C:\data\...`. Avoid WSL mixing.

- Linux permissions
  - Avoid `sudo` for Python packages. Use virtualenv/conda.

---

## 6. Expected Outputs
- Molecules: `experiments/generated_molecules.txt` or `experiments/final_validation/`
- Logs: `logs/training_full.log` shows per-epoch `d_loss` and `g_loss`
- Checkpoints: `experiments/checkpoints/`
- Success Criteria:
  - GAN completes 20 epochs
  - QSAR displays R² in logs
  - Toxicity model completes without import errors
  - Streamlit UI loads on port 8501 and displays molecules

---

## Compatibility Table (Key Packages)

- `PyTorch` — macOS Intel/ARM, Windows, Linux — fixed: CPU builds — v2.2.x
- `TorchVision` — macOS Intel/ARM, Windows, Linux — fixed: CPU builds — v0.17.x
- `Torch Geometric` — macOS Intel/ARM, Linux — Windows via Conda — v2.5.x
- `PennyLane` — macOS/Windows/Linux — autoray pinned — v0.38.0
- `autoray` — macOS/Windows/Linux — pinned to avoid API error — v0.6.11
- `Qiskit` — macOS/Windows/Linux — CPU only — v1.2.0
- `DeepChem` — macOS/Windows/Linux — dataset loaders validated — v2.8.0
- `RDKit` — macOS Intel/Ubuntu via pip, ARM/Windows via conda — latest
- `OpenBabel` — macOS via brew, Ubuntu via apt, Windows installer
- `Streamlit` — all OS — CORS disabled via config — v1.37.0

For any OS-specific build issues, prefer Conda environments and `conda-forge` packages.

---

## Notes
- The project uses `.streamlit/config.toml` to avoid CORS/XSRF issues and default to port `8501`.
- If `8501` is busy, start with `--server.port 8502`.

Ready to run end‑to‑end on supported systems.

# System Build and Deployment — Hybrid Quantum GAN Drug Discovery

Date: Tue Dec 2, 2025

This document describes how the project was conceived, implemented, validated, and deployed end-to-end. It consolidates architecture, environment setup, data handling, training, UI, monitoring, artifacts, and troubleshooting.

---

## 1. Vision & Scope

- Objective: Generate and evaluate drug-like molecules using a hybrid quantum-classical GAN, predict properties (QSAR), assess toxicity, estimate docking affinity, optimize via RL, and explain predictions — all with an interactive UI.
- Constraints: CPU-only macOS support; reproducible environment; robust logging and modularity.

---

## 2. Architecture Overview

- Generator: HQ-MolGAN with quantum VQC layer (`src/quantum/vqc_vvrq.py`), torch-integrated QNode (`interface="torch"`).
- Discriminator: MolGAN-style and transformer options (`src/models/gan/`).
- Properties (QSAR): RF/MLP/GNN models (`src/models/qsar/`).
- Toxicity: Multi-task classifier (`src/models/toxicity/`).
- Docking: AutoDock Vina integration (`src/models/docking/`).
- Reinforcement Learning: Reward functions + policy trainer (`src/models/rl/`).
- Explainability: SHAP-based analysis (`src/models/explainability/`).
- UI: Streamlit app (`src/ui/`) with feature pages.

Key orchestrators:
- `src/training/train_all_pipeline.py`: Full pipeline (GAN → QSAR → Toxicity).
- `scripts/generate_samples.py`: Generation utility.
- `src/evaluation/*`: Metrics for GAN/QSAR/Toxicity/Docking.

---

## 3. Environment & Compatibility

- Python: 3.11.7 (macOS Intel).
- Pinned versions: `torch==2.2.2`, `torch_geometric==2.5.3`, `pennylane==0.38.0`, `autoray==0.6.11`, `qiskit==1.2.0`, `deepchem==2.8.0`, `rdkit==2025.09.2`, `streamlit==1.37.0`.
- Streamlit CORS/XSRF disabled via `.streamlit/config.toml`.
- Validation artifact: `experiments/final_validation/compatibility_report.json`.

Setup (macOS zsh):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Data Pipeline

- Datasets: ESOL, FreeSolv, Tox21, ToxCast via DeepChem; unified CSVs for QSAR/tox tasks.
- Preprocessing: Tokenization, graph featurization, train/val/test splits, cache rebuild (~850 molecules, ≤9 atoms).
- Scripts:
  - `scripts/download_datasets.sh`
  - `scripts/preprocess_all.py`

---

## 5. Training Pipeline

Run full pipeline:
```bash
python -m src.training.train_all_pipeline
```
- Logs: `logs/training_full.log` (live progress)
- Checkpoints: `experiments/checkpoints/gan_epoch_*.pt`

Monitor training:
```bash
tail -f logs/training_full.log
```

GAN epochs produce generator/discriminator losses; upon completion, QSAR and Toxicity training steps run and log their metrics.

---

## 6. Generation, Evaluation, UI

Generate molecules:
```bash
python scripts/generate_samples.py --num_samples 10
```

Evaluate GAN:
```bash
python -m src.evaluation.eval_gan_metrics
```

Launch UI:
```bash
streamlit run src/ui/streamlit_app.py
# Open http://localhost:8501
```

Report generation:
```bash
python -m src.reports.pdf_report_generator
```

---

## 7. Final Validation Artifacts

Location: `experiments/final_validation/`
- `generated_smiles.json`: top-5 molecules (qed, logp, sa, smiles)
- `results_table.csv`: unified metrics (qsar_score, tox_prob, dock_affinity, reward)
- `summary.json`: validity summary (100% roundtrip-valid for 5)
- `compatibility_report.json`: environment and package compatibility
- `run_summary.txt`: human-readable overview (creation → deployment) and training status

---

## 8. Operational Notes & Fixes

- Quantum/Torch integration: Construct QNodes with `interface="torch"` in `vqc_vvrq.py` to ensure gradient flow.
- SSL downloads: Upgrade `certifi` and set `SSL_CERT_FILE` when DeepChem downloads fail.
- Minimal pipeline: Fixed RDKit descriptor usage and docking scores syntax in `scripts/end_to_end_minimal.py`.
- Streamlit: Disabled CORS/XSRF for local development.

---

## 9. Troubleshooting

Imports:
```bash
python -c "import sys; print(sys.executable)"
pip install --upgrade <pkg>
```

RDKit:
```bash
pip uninstall rdkit rdkit-pypi
pip install rdkit-pypi==2022.9.5
```

Quantum backends:
```bash
pip install pennylane==0.38.0 autoray==0.6.11
pip install qiskit==1.2.0 qiskit-aer==0.15.1
```

Memory:
- Reduce batch sizes in `configs/data.yaml`.

Docking:
```bash
brew install autodock-vina
vina --version
```

---

## 10. What Was Created

- Docs: `README.md`, `QUICKSTART.md`, `INSTALL_AND_RUN.md`, `PROJECT_SUMMARY.md`, and this `SYSTEM_BUILD_AND_DEPLOYMENT.md`.
- Configs: `configs/*.yaml` covering data, GAN, quantum, QSAR, toxicity, docking, UI.
- Modules: Full stack under `src/` including quantum layers, GAN, QSAR, toxicity, docking, RL, explainability, training, evaluation, UI, reports.
- Scripts: `scripts/*.py` and `.sh` for data, preprocessing, training, generation, UI.
- Artifacts: `experiments/` outputs and `logs/` for monitoring.

---

## 11. Deployment Summary

Local deployment:
1. Create venv and install requirements
2. Download and preprocess datasets
3. Run training pipeline (background or foreground)
4. Launch Streamlit UI and explore features
5. Generate reports and review artifacts in `experiments/`

Optional extension:
- Containerize with a minimal Dockerfile (CPU-only)
- Host Streamlit behind a reverse proxy with CORS disabled

---

## 12. Status & Next Steps

- Training is currently active; monitor via `logs/training_full.log`.
- After GAN completes, QSAR and Toxicity metrics will be appended to logs and artifacts.
- Recommend reviewing `experiments/final_validation/results_table.csv` and generating a PDF summary once all stages finish.

---

For quick start, see `INSTALL_AND_RUN.md` and `QUICKSTART.md`. For module-by-module status, see `PROJECT_SUMMARY.md`.

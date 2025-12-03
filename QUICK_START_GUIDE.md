# Quick Start Guide

Run the app with one command on **any system** (macOS, Linux, Windows):

## Option 1: Bash Script (macOS/Linux/WSL/Git Bash)

```bash
./QUICK_START.sh --ui
```

## Option 2: Manual (All platforms)

```bash
# macOS/Linux
python3 -m venv .venv && source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv .venv
.venv\Scripts\activate.bat

# Install and run
pip install -r requirements.txt
streamlit run src/ui/streamlit_app.py
```

## Access the App

Open: **http://localhost:8501**

## Zero CORS Issues

The project is pre-configured with:
- `enableCORS = false` in `.streamlit/config.toml`
- `enableXsrfProtection = false` for local development
- Works on all devices without network/firewall issues

## Generate Molecules

1. Go to "Generate Molecules" page
2. Set number of molecules (1-100)
3. Click "Generate Molecules"
4. View images, metrics, and copy SMILES

## Package for Distribution

```bash
# Exclude cache and virtual env
zip -r hqgan_project.zip . -x ".venv/*" "__pycache__/*" "*.pyc" "data/raw/*"
```

Recipients just unzip and run `QUICK_START.sh --ui`!

## Requirements

- Python 3.10 or 3.11
- 8GB+ RAM recommended
- No GPU required (CPU-only)

## Training (Optional)

```bash
python scripts/preprocess_all.py
python -m src.training.train_all_pipeline
```

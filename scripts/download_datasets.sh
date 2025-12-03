#!/bin/bash
# Download molecular datasets (QM9, ESOL, FreeSolv, PCBA, Tox21, ToxCast)
set -euo pipefail

# Ensure we run from project root
cd "$(dirname "$0")/.."

# Prefer python3
PY=python3
if ! command -v $PY >/dev/null 2>&1; then
    echo "python3 not found in PATH" >&2
    exit 1
fi

# Attempt to fix macOS SSL certs for Python downloads
if $PY -c "import certifi; print(certifi.where())" >/dev/null 2>&1; then
    export SSL_CERT_FILE="$($PY -c 'import certifi; print(certifi.where())')"
fi

echo "======================================"
echo "Downloading Molecular Datasets"
echo "======================================"

# Create directories
mkdir -p data/raw/{qm9,pc9,bioactivity,tfds,esol,freesolv,pcba,tox21,toxcast}
mkdir -p data/{interim,processed}

echo ""
echo "[1/6] Downloading QM9 via TFDS..."
$PY - <<'PYCODE'
import os, sys
os.makedirs('data/raw/tfds', exist_ok=True)
try:
    import tensorflow_datasets as tfds
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow-datasets==4.9.6'])
    import tensorflow_datasets as tfds

print('Requesting TFDS qm9...')
builder = tfds.builder('qm9', data_dir='data/raw/tfds')
builder.download_and_prepare()
print('✓ TFDS qm9 prepared at data/raw/tfds')
PYCODE

echo "Downloading QM9 (DeepChem loader cache)..."
$PY - <<'PYCODE'
import sys
sys.path.append('.')
from src.data import load_molecular_dataset
print('Loading QM9 via project loader (for local cache)...')
smiles, props = load_molecular_dataset('qm9', max_samples=None)
print(f'✓ QM9 loader returned {len(smiles)} molecules')
PYCODE

echo ""
echo "[2/6] Downloading ESOL/FreeSolv/PCBA/Tox21/ToxCast via DeepChem..."
$PY - <<'PYCODE'
import os
import json
import pandas as pd
import deepchem as dc

os.makedirs('data/raw/esol', exist_ok=True)
os.makedirs('data/raw/freesolv', exist_ok=True)
os.makedirs('data/raw/pcba', exist_ok=True)
os.makedirs('data/raw/tox21', exist_ok=True)
os.makedirs('data/raw/toxcast', exist_ok=True)

def save_dataset(ds, tasks, out_csv):
    X, y, w, ids = ds.X, ds.y, ds.w, ds.ids
    # Flatten multi-task labels if needed
    if y.ndim == 2 and y.shape[1] > 1:
        df = pd.DataFrame(y, columns=tasks)
    else:
        df = pd.DataFrame({'target': y.reshape(-1)})
    df.insert(0, 'smiles', ids)
    df.to_csv(out_csv, index=False)

# ESOL
try:
    tasks, (train, valid, test), _ = dc.molnet.load_delaney(featurizer='ECFP')
    save_dataset(train, tasks, 'data/raw/esol/esol_train.csv')
    save_dataset(valid, tasks, 'data/raw/esol/esol_valid.csv')
    save_dataset(test, tasks, 'data/raw/esol/esol_test.csv')
    print('✓ ESOL downloaded and saved')
except Exception as e:
    print('! ESOL download failed:', e)

# FreeSolv
try:
    tasks, (train, valid, test), _ = dc.molnet.load_freesolv(featurizer='ECFP')
    save_dataset(train, tasks, 'data/raw/freesolv/freesolv_train.csv')
    save_dataset(valid, tasks, 'data/raw/freesolv/freesolv_valid.csv')
    save_dataset(test, tasks, 'data/raw/freesolv/freesolv_test.csv')
    print('✓ FreeSolv downloaded and saved')
except Exception as e:
    print('! FreeSolv download failed:', e)

# PCBA (large)
try:
    tasks, (train, valid, test), _ = dc.molnet.load_pcba(featurizer='ECFP', split='random')
    save_dataset(train, tasks, 'data/raw/pcba/pcba_train.csv')
    save_dataset(valid, tasks, 'data/raw/pcba/pcba_valid.csv')
    save_dataset(test, tasks, 'data/raw/pcba/pcba_test.csv')
    with open('data/raw/pcba/tasks.json','w') as f:
        json.dump(tasks, f)
    print('✓ PCBA downloaded and saved')
except Exception as e:
    print('! PCBA download failed (may be very large):', e)

# Tox21
try:
    tasks, (train, valid, test), _ = dc.molnet.load_tox21(featurizer='ECFP')
    save_dataset(train, tasks, 'data/raw/tox21/tox21_train.csv')
    save_dataset(valid, tasks, 'data/raw/tox21/tox21_valid.csv')
    save_dataset(test, tasks, 'data/raw/tox21/tox21_test.csv')
    with open('data/raw/tox21/tasks.json','w') as f:
        json.dump(tasks, f)
    print('✓ Tox21 downloaded and saved')
except Exception as e:
    print('! Tox21 download failed:', e)

# ToxCast
try:
    tasks, (train, valid, test), _ = dc.molnet.load_toxcast(featurizer='ECFP')
    save_dataset(train, tasks, 'data/raw/toxcast/toxcast_train.csv')
    save_dataset(valid, tasks, 'data/raw/toxcast/toxcast_valid.csv')
    save_dataset(test, tasks, 'data/raw/toxcast/toxcast_test.csv')
    with open('data/raw/toxcast/tasks.json','w') as f:
        json.dump(tasks, f)
    print('✓ ToxCast downloaded and saved')
except Exception as e:
    print('! ToxCast download failed:', e)
PYCODE

echo "[3/6] Curated QM9 (optional)"
echo "Attempting to fetch curated QM9 metadata..."
$PY - <<'PYCODE'
import os, requests
os.makedirs('data/raw/qm9/curated', exist_ok=True)
url = 'https://raw.githubusercontent.com/moldis-group/curatedQM9/master/data/curated_qm9.json'
try:
    r = requests.get(url, timeout=60)
    if r.ok:
        open('data/raw/qm9/curated/curated_qm9.json','wb').write(r.content)
        print('✓ curated_qm9.json downloaded')
    else:
        print('curated QM9 not available:', r.status_code)
except Exception as e:
    print('curated QM9 fetch failed (optional):', e)
PYCODE

echo "[4/6] Downloading QM8/GDB8 archive (from provided URL)..."
echo "URL: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb8.tar.gz"
mkdir -p data/raw/gdb8
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 -o data/raw/gdb8/gdb8.tar.gz \
    https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb8.tar.gz || echo "Skipping gdb8.tar.gz"
elif command -v wget >/dev/null 2>&1; then
  wget -O data/raw/gdb8/gdb8.tar.gz https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb8.tar.gz || echo "Skipping gdb8.tar.gz"
fi

echo "[5/6] Preparing unified CSVs for QSAR/Toxicity..."
$PY scripts/preprocess_molnet.py || true

echo ""
echo "[6/6] Preparing datasets..."
echo "✓ Datasets ready!"
echo ""
echo "======================================"
echo "Dataset Download Complete!"
echo "======================================"

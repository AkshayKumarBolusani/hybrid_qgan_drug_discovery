#!/usr/bin/env python3
"""Unify MoleculeNet datasets into CSVs for QSAR and Toxicity.

Outputs under data/processed/molecule_datasets/:
- esol.csv (smiles, target)
- freesolv.csv (smiles, target)
- pcba.csv (smiles, <pcba tasks...>)
- tox21.csv (smiles, <tox21 tasks...>)
- toxcast.csv (smiles, <toxcast tasks...>)
- combined_qsar.csv (concatenated ESOL + FreeSolv with dataset tag)
- combined_toxicity.csv (wide multi-label for tox21)
"""

import os
import json
from pathlib import Path
import pandas as pd

RAW = Path('data/raw')
OUT = Path('data/processed/molecule_datasets')
OUT.mkdir(parents=True, exist_ok=True)


def load_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


# ESOL
esol_train = load_if_exists(RAW / 'esol/esol_train.csv')
esol_valid = load_if_exists(RAW / 'esol/esol_valid.csv')
esol_test  = load_if_exists(RAW / 'esol/esol_test.csv')
if esol_train is not None and esol_valid is not None and esol_test is not None:
    esol = pd.concat([esol_train, esol_valid, esol_test], ignore_index=True)
    # Normalize columns
    if 'target' not in esol.columns:
        # In DeepChem ESOL, the single task is often named 'measured log solubility in mols per litre'
        # If present, rename to target
        for col in esol.columns:
            if col not in ('smiles', 'id') and esol[col].dtype != 'O':
                esol = esol.rename(columns={col: 'target'})
                break
    esol[['smiles', 'target']].to_csv(OUT / 'esol.csv', index=False)
    tmp = esol[['smiles', 'target']].copy()
    tmp['dataset'] = 'esol'
    combined_qsar = tmp
else:
    combined_qsar = pd.DataFrame(columns=['smiles','target','dataset'])

# FreeSolv
fs_train = load_if_exists(RAW / 'freesolv/freesolv_train.csv')
fs_valid = load_if_exists(RAW / 'freesolv/freesolv_valid.csv')
fs_test  = load_if_exists(RAW / 'freesolv/freesolv_test.csv')
if fs_train is not None and fs_valid is not None and fs_test is not None:
    fs = pd.concat([fs_train, fs_valid, fs_test], ignore_index=True)
    if 'target' not in fs.columns:
        for col in fs.columns:
            if col not in ('smiles', 'id') and fs[col].dtype != 'O':
                fs = fs.rename(columns={col: 'target'})
                break
    fs[['smiles', 'target']].to_csv(OUT / 'freesolv.csv', index=False)
    tmp = fs[['smiles', 'target']].copy()
    tmp['dataset'] = 'freesolv'
    combined_qsar = pd.concat([combined_qsar, tmp], ignore_index=True)

# PCBA
pcba_train = load_if_exists(RAW / 'pcba/pcba_train.csv')
pcba_valid = load_if_exists(RAW / 'pcba/pcba_valid.csv')
pcba_test  = load_if_exists(RAW / 'pcba/pcba_test.csv')
if pcba_train is not None and pcba_valid is not None and pcba_test is not None:
    pcba = pd.concat([pcba_train, pcba_valid, pcba_test], ignore_index=True)
    pcba.to_csv(OUT / 'pcba.csv', index=False)

# Tox21
tox21_train = load_if_exists(RAW / 'tox21/tox21_train.csv')
tox21_valid = load_if_exists(RAW / 'tox21/tox21_valid.csv')
tox21_test  = load_if_exists(RAW / 'tox21/tox21_test.csv')
if tox21_train is not None and tox21_valid is not None and tox21_test is not None:
    tox21 = pd.concat([tox21_train, tox21_valid, tox21_test], ignore_index=True)
    tox21.to_csv(OUT / 'tox21.csv', index=False)
    # Save combined toxicity as wide table
    tox_cols = [c for c in tox21.columns if c not in ('smiles','id')]
    tox21[['smiles', *tox_cols]].to_csv(OUT / 'combined_toxicity.csv', index=False)

# ToxCast
toxcast_train = load_if_exists(RAW / 'toxcast/toxcast_train.csv')
toxcast_valid = load_if_exists(RAW / 'toxcast/toxcast_valid.csv')
toxcast_test  = load_if_exists(RAW / 'toxcast/toxcast_test.csv')
if toxcast_train is not None and toxcast_valid is not None and toxcast_test is not None:
    toxcast = pd.concat([toxcast_train, toxcast_valid, toxcast_test], ignore_index=True)
    toxcast.to_csv(OUT / 'toxcast.csv', index=False)

# Combined QSAR
if not combined_qsar.empty:
    combined_qsar.to_csv(OUT / 'combined_qsar.csv', index=False)

print('✓ Unified datasets written to', OUT)

"""Quick training script using prepared CSV files."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("=" * 80)
print("QUICK TRAINING - Using Prepared CSV Files")
print("=" * 80)

# Paths
data_dir = Path("data/processed/molecule_datasets")
checkpoint_dir = Path("experiments/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# ====================
# 1. TRAIN QSAR MODEL
# ====================
print("\n[1/2] Training QSAR Model...")
try:
    qsar_df = pd.read_csv(data_dir / "combined_qsar.csv")
    print(f"  Loaded {len(qsar_df)} samples")
    
    # Extract SMILES and targets
    smiles = qsar_df['smiles'].tolist()
    
    # Use 'target' column (solubility values)
    y = qsar_df['target'].values
    
    # Generate fingerprints
    print("  Generating fingerprints...")
    from src.features.fingerprint_utils import batch_smiles_to_fingerprints
    X = np.array(batch_smiles_to_fingerprints(smiles, fp_type='morgan', n_bits=2048))
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=0)
    print("  Training Random Forest...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"  Train R²: {train_score:.3f}")
    print(f"  Test R²: {test_score:.3f}")
    
    # Save
    qsar_path = checkpoint_dir / "qsar_rf.pkl"
    with open(qsar_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print(f"  ✓ Saved to {qsar_path}")
    
except Exception as e:
    print(f"  ❌ QSAR training failed: {e}")

# ==========================
# 2. TRAIN TOXICITY MODEL
# ==========================
print("\n[2/2] Training Toxicity Model...")
try:
    tox_df = pd.read_csv(data_dir / "combined_toxicity.csv")
    print(f"  Loaded {len(tox_df)} samples")
    
    # Extract SMILES
    smiles = tox_df['smiles'].tolist()
    
    # Extract toxicity labels (12 tasks from Tox21)
    tox_columns = [col for col in tox_df.columns if col.startswith('NR-') or col.startswith('SR-')]
    print(f"  Found {len(tox_columns)} toxicity tasks")
    
    # Generate fingerprints
    print("  Generating fingerprints...")
    from src.features.fingerprint_utils import batch_smiles_to_fingerprints
    X = np.array(batch_smiles_to_fingerprints(smiles, fp_type='morgan', n_bits=2048))
    
    # Get labels
    y = tox_df[tox_columns].fillna(0).values  # Fill NaN with 0 (non-toxic)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train multi-task model
    print("  Training multi-task Random Forest...")
    from src.models.toxicity.tox_classifiers import SKLearnToxicityModel
    
    tox_model = SKLearnToxicityModel(num_tasks=len(tox_columns))
    tox_model.fit(X_train, y_train)
    
    # Evaluate (average accuracy across tasks)
    y_pred_train = tox_model.predict(X_train)
    y_pred_test = tox_model.predict(X_test)
    
    train_acc = np.mean([(y_train[:, i] == (y_pred_train[:, i] > 0.5)).mean() for i in range(len(tox_columns))])
    test_acc = np.mean([(y_test[:, i] == (y_pred_test[:, i] > 0.5)).mean() for i in range(len(tox_columns))])
    
    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    
    # Save
    tox_path = checkpoint_dir / "toxicity_rf.pkl"
    with open(tox_path, 'wb') as f:
        pickle.dump(tox_model, f)
    print(f"  ✓ Saved to {tox_path}")
    
except Exception as e:
    print(f"  ❌ Toxicity training failed: {e}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print("\nTrained models:")
print(f"  - QSAR: experiments/checkpoints/qsar_rf.pkl")
print(f"  - Toxicity: experiments/checkpoints/toxicity_rf.pkl")
print("\nYou can now use the QSAR & Toxicity page in the UI!")

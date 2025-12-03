#!/usr/bin/env python3
"""Verify project setup and dependencies."""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (need 3.10+)")
        return False

def check_imports():
    """Check if all required packages can be imported."""
    packages = {
        'torch': 'PyTorch',
        'pennylane': 'PennyLane',
        'qiskit': 'Qiskit',
        'rdkit': 'RDKit',
        'deepchem': 'DeepChem',
        'sklearn': 'scikit-learn',
        'xgboost': 'XGBoost',
        'streamlit': 'Streamlit',
        'shap': 'SHAP',
        'reportlab': 'ReportLab',
        'plotly': 'Plotly',
        'yaml': 'PyYAML'
    }
    
    results = {}
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✅ {name}")
            results[name] = True
        except ImportError:
            print(f"❌ {name} - Not installed")
            results[name] = False
    
    return all(results.values())

def check_directories():
    """Check if required directories exist."""
    dirs = [
        'configs',
        'src',
        'src/utils',
        'src/data',
        'src/features',
        'src/quantum',
        'src/models',
        'src/training',
        'src/evaluation',
        'src/ui',
        'src/reports',
        'scripts',
        'data',
        'experiments',
        'logs'
    ]
    
    all_exist = True
    for dir_path in dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - Missing")
            all_exist = False
    
    return all_exist

def check_config_files():
    """Check if configuration files exist."""
    configs = [
        'configs/project.yaml',
        'configs/data.yaml',
        'configs/gan.yaml',
        'configs/quantum.yaml',
        'configs/qsar.yaml',
        'configs/tox_admet.yaml',
        'configs/docking.yaml'
    ]
    
    all_exist = True
    for config in configs:
        if Path(config).exists():
            print(f"✅ {config}")
        else:
            print(f"❌ {config} - Missing")
            all_exist = False
    
    return all_exist

def check_scripts():
    """Check if utility scripts exist and are executable."""
    scripts = [
        'scripts/download_datasets.sh',
        'scripts/preprocess_all.py',
        'scripts/run_full_training.sh',
        'scripts/generate_samples.py',
        'scripts/run_streamlit.sh'
    ]
    
    all_ok = True
    for script in scripts:
        path = Path(script)
        if path.exists():
            if path.suffix == '.sh':
                import os
                if os.access(str(path), os.X_OK):
                    print(f"✅ {script} (executable)")
                else:
                    print(f"⚠️  {script} (not executable - run: chmod +x {script})")
            else:
                print(f"✅ {script}")
        else:
            print(f"❌ {script} - Missing")
            all_ok = False
    
    return all_ok

def main():
    """Run all checks."""
    print("=" * 60)
    print("HYBRID QUANTUM GAN DRUG DISCOVERY - SETUP VERIFICATION")
    print("=" * 60)
    
    print("\n[1/5] Checking Python Version...")
    python_ok = check_python_version()
    
    print("\n[2/5] Checking Package Imports...")
    imports_ok = check_imports()
    
    print("\n[3/5] Checking Directory Structure...")
    dirs_ok = check_directories()
    
    print("\n[4/5] Checking Configuration Files...")
    configs_ok = check_config_files()
    
    print("\n[5/5] Checking Utility Scripts...")
    scripts_ok = check_scripts()
    
    print("\n" + "=" * 60)
    if python_ok and imports_ok and dirs_ok and configs_ok and scripts_ok:
        print("✅ ALL CHECKS PASSED - System ready to use!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. ./scripts/download_datasets.sh")
        print("  2. python scripts/preprocess_all.py")
        print("  3. ./scripts/run_full_training.sh")
        print("  4. python scripts/generate_samples.py --num_samples 10")
        print("  5. ./scripts/run_streamlit.sh")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please install missing dependencies")
        print("=" * 60)
        print("\nRun: pip install -r requirements.txt")
        return 1

if __name__ == '__main__':
    sys.exit(main())

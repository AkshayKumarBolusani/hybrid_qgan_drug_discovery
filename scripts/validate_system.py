#!/usr/bin/env python3
import os, sys, platform, json
from pathlib import Path

report = {}
root = Path(__file__).resolve().parents[1]
os.chdir(root)

# Detect OS and architecture
report['system'] = {
  'os': platform.system(),
  'version': platform.version(),
  'machine': platform.machine(),
  'python': sys.version,
}

# Streamlit CORS config
cors_ok = False
cfg_path = root/".streamlit"/"config.toml"
if cfg_path.exists():
  txt = cfg_path.read_text()
  cors_ok = ('server.enableCORS = false' in txt) and ('server.enableXsrfProtection = false' in txt)
report['streamlit'] = {'config': str(cfg_path), 'cors_safe': cors_ok}

# Package versions
pkgs = {
  'torch': None,
  'torchvision': None,
  'torch_geometric': None,
  'pennylane': None,
  'autoray': None,
  'qiskit': None,
  'deepchem': None,
  'rdkit': None,
  'streamlit': None,
}
for name in list(pkgs.keys()):
  try:
    m = __import__(name)
    v = getattr(m, '__version__', 'unknown')
    pkgs[name] = v
  except Exception as e:
    pkgs[name] = f'not-installed: {e}'
report['packages'] = pkgs

# Compatibility table (static rules + detected)
compat = []
compat_rules = {
  'torch': 'CPU builds; macOS/Windows/Linux OK',
  'torch_geometric': 'Windows via Conda preferred; macOS/Linux OK',
  'pennylane': 'Pin autoray<=0.6.11',
  'autoray': 'Use 0.6.11 with PennyLane 0.38',
  'qiskit': 'CPU simulators OK on all OS',
  'deepchem': 'OK; if SSL fails set SSL_CERT_FILE',
  'rdkit': 'Conda on macOS ARM/Windows',
  'streamlit': 'CORS disabled in config',
}
for pkg, ver in pkgs.items():
  compat.append({
    'package': pkg,
    'version': ver,
    'os_support': 'macOS Intel/ARM, Windows, Ubuntu',
    'fix': compat_rules.get(pkg, ''),
  })
report['compatibility_table'] = compat

# UI import checks
ui_imports = []
for p in (root/"app").glob("**/*.py"):
  try:
    mod = p.relative_to(root).with_suffix("").as_posix().replace('/', '.')
    __import__(mod)
    ui_imports.append({'module': mod, 'status': 'ok'})
  except Exception as e:
    ui_imports.append({'module': str(p), 'status': f'fail: {e}'})
report['ui_imports'] = ui_imports

# Models/pipelines import
modules = [
  'src.quantum.vqc_vvrq',
  'src.models.gan.generator_hqmolgan',
  'src.training.train_qsar',
  'src.training.train_toxicity',
]
mods = []
for m in modules:
  try:
    __import__(m)
    mods.append({'module': m, 'status': 'ok'})
  except Exception as e:
    mods.append({'module': m, 'status': f'fail: {e}'})
report['module_imports'] = mods

out = root/"experiments"/"final_validation"
out.mkdir(parents=True, exist_ok=True)
(out/"compatibility_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

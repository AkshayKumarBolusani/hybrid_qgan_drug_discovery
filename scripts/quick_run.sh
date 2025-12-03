#!/usr/bin/env zsh
set -e

# Minimal one-liner style quick run
# Usage: ./scripts/quick_run.sh

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
PY_VER=${PY_VER:-3.11}
PORT=${PORT:-8501}

printf "\n== Quick Run: Hybrid Quantum GAN Drug Discovery ==\n"
printf "Project: $PROJECT_DIR\nPort: $PORT\nPython: $PY_VER\n\n"

# 1) Create venv if missing
if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
  python$PY_VER -m venv "$PROJECT_DIR/.venv"
fi
source "$PROJECT_DIR/.venv/bin/activate"

# 2) Install core deps (fast path) if Streamlit missing
python - <<'PY'
import importlib, sys
pkgs = ["streamlit"]
missing = []
for p in pkgs:
    try:
        importlib.import_module(p)
    except Exception:
        missing.append(p)
if missing:
    print("Installing:", ", ".join(missing))
else:
    print("All core packages present")
PY

if ! python -c "import streamlit" >/dev/null 2>&1; then
  pip install -r requirements.txt
fi

# 3) Launch Streamlit headless with CORS disabled
streamlit run src/ui/streamlit_app.py --server.port=$PORT --server.headless=true

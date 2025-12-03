@echo off
REM Universal Quick Start for Hybrid Quantum GAN Drug Discovery - Windows
REM Works on Windows CMD and PowerShell

echo.
echo 🚀 Hybrid Quantum GAN Drug Discovery - Quick Start
echo ==================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VERSION=%%i
echo ✓ Found Python %PY_VERSION%

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Setting up directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\interim" mkdir data\interim
if not exist "experiments\checkpoints" mkdir experiments\checkpoints
if not exist "experiments\final_validation" mkdir experiments\final_validation
if not exist "logs" mkdir logs

echo.
echo ✅ Setup complete!
echo.
echo 🎯 Next steps:
echo    1. Launch UI:    streamlit run src/ui/streamlit_app.py
echo    2. Or run:       QUICK_START.bat --ui
echo.

REM If --ui flag is passed, launch the UI
if "%1"=="--ui" (
    echo 🌐 Launching Streamlit UI...
    streamlit run src/ui/streamlit_app.py --server.port=8501 --server.headless=true
) else if "%1"=="/ui" (
    echo 🌐 Launching Streamlit UI...
    streamlit run src/ui/streamlit_app.py --server.port=8501 --server.headless=true
)

if not "%1"=="--ui" if not "%1"=="/ui" pause

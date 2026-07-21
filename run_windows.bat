@echo off
REM ==========================================================================
REM  Pigment Inverse Designer - Windows launcher
REM  1) creates/uses a local venv  2) installs deps  3) downloads data
REM  4) trains the model if needed  5) launches Streamlit
REM ==========================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\" (
    echo [1/5] Creating virtual environment (.venv) ...
    python -m venv .venv
)

echo [2/5] Activating environment and installing dependencies ...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip >nul
pip install -r requirements.txt

echo [3/5] Downloading dataset (falls back to sample on failure) ...
python -m scripts.download_data

if not exist "models\absorption_model.joblib" (
    echo [4/5] Training baseline model ...
    python -m scripts.train_model
) else (
    echo [4/5] Existing model found, skipping training.
)

echo [5/5] Launching Streamlit app ...
streamlit run app.py

endlocal

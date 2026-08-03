@echo off
REM Run dataset preprocessing
REM Usage: run_preprocessing.bat [--dry-run]

echo ========================================
echo Thermal Image Preprocessing Pipeline
echo ========================================
echo.

cd /d "%~dp0\.."

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run preprocessing
python preprocessing/preprocess.py --config configs/preprocessing.yaml %*

echo.
echo Preprocessing complete!
pause

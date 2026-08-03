@echo off
REM Run dataset deduplication
REM Usage: run_deduplication.bat [--dry-run]

echo ========================================
echo Thermal Image Dataset Deduplication
echo ========================================
echo.

cd /d "%~dp0\.."

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run deduplication
python -m preprocessing.deduplicate --config configs/deduplication.yaml %*

echo.
echo Deduplication complete!
pause

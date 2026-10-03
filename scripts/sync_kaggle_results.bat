@echo off
REM Windows batch script to sync Kaggle results
REM Usage: sync_kaggle_results.bat

echo ============================================================
echo Kaggle Results Sync - Windows
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python or add it to your PATH
    pause
    exit /b 1
)

REM Get script directory
set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..

echo Choose sync method:
echo   1. Download using Kaggle CLI (requires kaggle package)
echo   2. Use manually downloaded zip file
echo.
set /p CHOICE="Enter choice (1 or 2): "

if "%CHOICE%"=="1" (
    set /p NOTEBOOK_URL="Enter Kaggle notebook URL or username/notebook: "
    python "%SCRIPT_DIR%sync_kaggle_results.py" --notebook-url "%NOTEBOOK_URL%"
) else if "%CHOICE%"=="2" (
    set /p ZIP_PATH="Enter path to downloaded zip file: "
    python "%SCRIPT_DIR%sync_kaggle_results.py" --manual --kaggle-zip "%ZIP_PATH%"
) else (
    echo Invalid choice
    pause
    exit /b 1
)

echo.
pause

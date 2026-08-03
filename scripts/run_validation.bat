@echo off
REM Validate processed datasets
REM This script validates that preprocessing completed successfully

echo ======================================================================
echo PREPROCESSING VALIDATION
echo ======================================================================
echo.

python preprocessing\validate.py --config configs\preprocessing.yaml

echo.
echo Validation complete. Check reports/preprocessing/ for results.
echo.

pause

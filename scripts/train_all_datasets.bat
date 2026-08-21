@echo off
REM Train CAE models for all 4 datasets sequentially

echo ================================================================================
echo TRAINING CAE MODELS FOR ALL DATASETS
echo ================================================================================
echo.
echo This will train 4 separate CAE models sequentially:
echo   1. Chilli_Leaves
echo   2. Okra  
echo   3. Paddy_Leaves
echo   4. No_Mixed
echo.
echo Each model will be saved to: checkpoints\cae\[dataset_name]\best_model.pth
echo.
pause

echo.
echo ================================================================================
echo [1/4] Training Chilli_Leaves model...
echo ================================================================================
python scripts\train_cae.py --dataset Chilli_Leaves
if errorlevel 1 (
    echo ✗ Chilli_Leaves training failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [2/4] Training Okra model...
echo ================================================================================
python scripts\train_cae.py --dataset Okra
if errorlevel 1 (
    echo ✗ Okra training failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [3/4] Training Paddy_Leaves model...
echo ================================================================================
python scripts\train_cae.py --dataset Paddy_Leaves
if errorlevel 1 (
    echo ✗ Paddy_Leaves training failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [4/4] Training No_Mixed model...
echo ================================================================================
python scripts\train_cae.py --dataset No_Mixed
if errorlevel 1 (
    echo ✗ No_Mixed training failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo ALL MODELS TRAINED SUCCESSFULLY!
echo ================================================================================
echo.
echo Models saved to:
echo   - checkpoints\cae\Chilli_Leaves\best_model.pth
echo   - checkpoints\cae\Okra\best_model.pth
echo   - checkpoints\cae\Paddy_Leaves\best_model.pth
echo   - checkpoints\cae\No_Mixed\best_model.pth
echo.
pause

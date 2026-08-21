@echo off
REM Resume training for incomplete CAE models (Chilli_Leaves and No_Mixed)

echo ================================================================================
echo RESUMING CAE TRAINING FOR INCOMPLETE DATASETS
echo ================================================================================
echo.
echo Training status:
echo   ✓ Okra - Complete (epoch 50)
echo   ✓ Paddy_Leaves - Complete (epoch 60)
echo   ⚠ Chilli_Leaves - Incomplete (stopped at epoch 40)
echo   ⚠ No_Mixed - Incomplete (stopped at epoch 40)
echo.
echo This script will resume training from the last checkpoint:
echo   1. Chilli_Leaves (resume from epoch 40)
echo   2. No_Mixed (resume from epoch 40)
echo.
pause

echo.
echo ================================================================================
echo [1/2] Resuming Chilli_Leaves training from epoch 40...
echo ================================================================================
python scripts\train_cae.py --dataset Chilli_Leaves --resume checkpoints\cae\Chilli_Leaves\checkpoint_epoch_40.pth
if errorlevel 1 (
    echo ✗ Chilli_Leaves training failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [2/2] Resuming No_Mixed training from epoch 40...
echo ================================================================================
python scripts\train_cae.py --dataset No_Mixed --resume checkpoints\cae\No_Mixed\checkpoint_epoch_40.pth
if errorlevel 1 (
    echo ✗ No_Mixed training failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo TRAINING RESUMED AND COMPLETED SUCCESSFULLY!
echo ================================================================================
echo.
echo All 4 models are now complete:
echo   ✓ checkpoints\cae\Chilli_Leaves\best_model.pth
echo   ✓ checkpoints\cae\Okra\best_model.pth
echo   ✓ checkpoints\cae\Paddy_Leaves\best_model.pth
echo   ✓ checkpoints\cae\No_Mixed\best_model.pth
echo.
pause

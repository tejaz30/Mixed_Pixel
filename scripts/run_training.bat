@echo off
REM Batch script to train CAE models for all datasets
REM This will train 4 separate models, one for each dataset

echo ================================================================================
echo TRAINING CAE MODELS FOR ALL DATASETS
echo ================================================================================
echo.
echo This will train 4 separate CAE models:
echo   1. Chilli_Leaves (1,454 images)
echo   2. Okra (501 images)
echo   3. Paddy_Leaves (636 images)
echo   4. No_Mixed (2,068 images)
echo.
echo Training configuration:
echo   - Max epochs: 500 per model
echo   - Early stopping: patience=5
echo   - Batch size: 32
echo   - Learning rate: 0.001
echo.
echo Models will be saved to: checkpoints\cae\[dataset_name]\
echo.
pause

python scripts\train_multiple_cae.py

echo.
echo ================================================================================
echo TRAINING COMPLETE
echo ================================================================================
pause

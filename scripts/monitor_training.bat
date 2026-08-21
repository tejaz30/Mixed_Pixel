@echo off
REM Monitor training progress for all datasets

:loop
cls
echo ================================================================================
echo CAE TRAINING PROGRESS MONITOR
echo ================================================================================
echo Current Time: %date% %time%
echo.

REM Check which datasets are being trained
echo Checking training status...
echo.

if exist "logs\cae\Chilli_Leaves\train.log" (
    echo [1] CHILLI_LEAVES:
    echo    Last 3 log lines:
    powershell -Command "Get-Content 'logs\cae\Chilli_Leaves\train.log' -Tail 3 | ForEach-Object { Write-Host '    ' $_ }"
    if exist "checkpoints\cae\Chilli_Leaves\best_model.pth" (
        echo    ✓ Model saved
    )
    echo.
)

if exist "logs\cae\Okra\train.log" (
    echo [2] OKRA:
    echo    Last 3 log lines:
    powershell -Command "Get-Content 'logs\cae\Okra\train.log' -Tail 3 | ForEach-Object { Write-Host '    ' $_ }"
    if exist "checkpoints\cae\Okra\best_model.pth" (
        echo    ✓ Model saved
    )
    echo.
)

if exist "logs\cae\Paddy_Leaves\train.log" (
    echo [3] PADDY_LEAVES:
    echo    Last 3 log lines:
    powershell -Command "Get-Content 'logs\cae\Paddy_Leaves\train.log' -Tail 3 | ForEach-Object { Write-Host '    ' $_ }"
    if exist "checkpoints\cae\Paddy_Leaves\best_model.pth" (
        echo    ✓ Model saved
    )
    echo.
)

if exist "logs\cae\No_Mixed\train.log" (
    echo [4] NO_MIXED:
    echo    Last 3 log lines:
    powershell -Command "Get-Content 'logs\cae\No_Mixed\train.log' -Tail 3 | ForEach-Object { Write-Host '    ' $_ }"
    if exist "checkpoints\cae\No_Mixed\best_model.pth" (
        echo    ✓ Model saved
    )
    echo.
)

echo ================================================================================
echo Press Ctrl+C to exit, or wait 10 seconds for auto-refresh...
echo ================================================================================

timeout /t 10 /nobreak >nul
goto loop

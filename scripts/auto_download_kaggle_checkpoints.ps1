# Auto-download Kaggle checkpoints while training runs
# Usage: powershell -File scripts/auto_download_kaggle_checkpoints.ps1 <your-kaggle-username> <notebook-name>

param(
    [Parameter(Mandatory=$true)]
    [string]$Username,
    
    [Parameter(Mandatory=$true)]
    [string]$NotebookName,
    
    [int]$CheckIntervalMinutes = 30
)

$ErrorActionPreference = "SilentlyContinue"

# Destination directory
$DestDir = "checkpoints\kl\Chilli_Leaves"
New-Item -ItemType Directory -Force -Path $DestDir | Out-Null

Write-Host "=" * 70
Write-Host "Kaggle Checkpoint Auto-Downloader"
Write-Host "=" * 70
Write-Host "Monitoring: $Username/$NotebookName"
Write-Host "Check interval: $CheckIntervalMinutes minutes"
Write-Host "Destination: $DestDir"
Write-Host "=" * 70
Write-Host ""
Write-Host "Press Ctrl+C to stop monitoring"
Write-Host ""

$iteration = 0

while ($true) {
    $iteration++
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    Write-Host "[$timestamp] Check #$iteration - Attempting download..."
    
    try {
        # Try to download notebook outputs
        $output = kaggle kernels output "$Username/$NotebookName" -p $DestDir 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Download successful!" -ForegroundColor Green
            
            # List downloaded files
            $files = Get-ChildItem -Path $DestDir -Recurse -File
            if ($files.Count -gt 0) {
                Write-Host "  Downloaded files:" -ForegroundColor Cyan
                foreach ($file in $files) {
                    $size = "{0:N2}" -f ($file.Length / 1MB)
                    Write-Host "    - $($file.Name) ($size MB)" -ForegroundColor Cyan
                }
            }
        } else {
            Write-Host "  ⚠ No outputs available yet (session may not be committed)" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "  ⚠ Error: $_" -ForegroundColor Yellow
    }
    
    # Wait before next check
    $waitSeconds = $CheckIntervalMinutes * 60
    Write-Host ""
    Write-Host "  Waiting $CheckIntervalMinutes minutes until next check..." -ForegroundColor Gray
    Write-Host ""
    
    Start-Sleep -Seconds $waitSeconds
}

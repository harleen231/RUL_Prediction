# AeroGuard Dashboard PowerShell Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting AeroGuard Aerospace Degradation Dashboard... " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$pyPath = "C:\Users\DELL\AppData\Local\Programs\Python\Python312\python.exe"

if (Test-Path $pyPath) {
    & $pyPath -m streamlit run dashboard/app.py
} else {
    python -m streamlit run dashboard/app.py
}

@echo off
title AeroGuard Dashboard Launcher
echo ========================================================
echo   Starting AeroGuard Aerospace Degradation Dashboard...
echo ========================================================
echo.

cd /d "%~dp0"

set "PY_EXE=C:\Users\DELL\AppData\Local\Programs\Python\Python312\python.exe"

if exist "%PY_EXE%" (
    echo Using Python at: %PY_EXE%
    "%PY_EXE%" -m streamlit run dashboard/app.py
) else (
    echo Searching for system Python...
    python -m streamlit run dashboard/app.py
)

pause

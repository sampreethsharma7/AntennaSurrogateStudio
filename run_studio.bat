@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "STUDIO_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%STUDIO_PYTHON%" (
    echo The local environment is missing. Run setup_windows.bat first.
    pause
    exit /b 1
)

"%STUDIO_PYTHON%" app.py
if errorlevel 1 pause

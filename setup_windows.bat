@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo Antenna Surrogate Studio - Windows Setup
echo ============================================================
echo.
echo This setup will:
echo   1. Find Python 3.10 or newer
echo   2. Create a local .venv environment
echo   3. Install required packages from requirements.txt
echo.

set "PYTHON_CMD="

py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    goto python_found
)

py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.10"
    goto python_found
)

python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto python_found
)

echo Python 3.10+ was not found.
echo Trying to install Python 3.11 automatically using winget...
winget --version >nul 2>&1
if errorlevel 1 goto python_missing
winget install -e --id Python.Python.3.11

py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    goto python_found
)

:python_missing
echo Please install Python 3.10 or newer from python.org, enable Add python.exe to PATH, then run setup_windows.bat again.
pause
exit /b 1

:python_found
echo Python found:
%PYTHON_CMD% --version
echo.

if exist ".venv\Scripts\python.exe" (
    echo Existing .venv environment found. It will be reused.
) else (
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo Failed to create .venv.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed. Check your internet connection and rerun setup_windows.bat.
    pause
    exit /b 1
)

echo.
echo Setup complete. Double-click run_app.bat to launch Antenna Surrogate Studio.
pause

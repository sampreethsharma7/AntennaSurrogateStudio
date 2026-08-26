@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo ============================================================
echo   Antenna Surrogate Studio - first-time setup
echo ============================================================
echo.

set "PYTHON_CMD="

py -3.12 -c "import sys, struct; raise SystemExit(0 if sys.version_info[:2] in ((3, 11), (3, 12), (3, 13)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.12"
if defined PYTHON_CMD goto python_found

py -3.11 -c "import sys, struct; raise SystemExit(0 if sys.version_info[:2] in ((3, 11), (3, 12), (3, 13)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.11"
if defined PYTHON_CMD goto python_found

py -3.13 -c "import sys, struct; raise SystemExit(0 if sys.version_info[:2] in ((3, 11), (3, 12), (3, 13)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.13"
if defined PYTHON_CMD goto python_found

python -c "import sys, struct; raise SystemExit(0 if sys.version_info[:2] in ((3, 11), (3, 12), (3, 13)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=python"
if defined PYTHON_CMD goto python_found

echo Python was not found. Attempting to install Python 3.12 with winget...
winget --version >nul 2>&1
if errorlevel 1 goto python_missing

winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
py -3.12 -c "import sys, struct; raise SystemExit(0 if sys.version_info[:2] in ((3, 11), (3, 12), (3, 13)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.12"
if not defined PYTHON_CMD goto python_missing

:python_found
echo Using:
%PYTHON_CMD% --version
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating the private application environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto setup_failed
)

echo Installing the streamlined desktop requirements...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 goto setup_failed
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto setup_failed

echo Checking the installed desktop and model dependencies...
".venv\Scripts\python.exe" -c "import tkinter, customtkinter, numpy, sklearn, scipy, joblib, xgboost"
if errorlevel 1 goto setup_failed

where ollama >nul 2>&1
if errorlevel 1 (
    echo.
    echo SnowBuddy local AI is optional and Ollama was not detected.
    echo The Studio will show the correct Windows, macOS, or Linux setup link.
) else (
    echo.
    echo Ollama detected. Choose Standard or Lightweight in SnowBuddy settings.
)

echo.
echo Setup complete.
if /I "%~1"=="/launch" exit /b 0
echo Double-click "Start Antenna Surrogate Studio.bat" to launch.
pause
exit /b 0

:python_missing
echo.
echo Python could not be installed automatically.
echo Install 64-bit Python 3.11, 3.12, or 3.13 from python.org, then run this setup again.
if /I "%~1"=="/launch" exit /b 1
pause
exit /b 1

:setup_failed
echo.
echo Setup failed. Check your internet connection and Python installation.
echo If Windows reports an Application Control policy, keep the Studio in a
echo writable Documents folder or ask the computer administrator to allow it.
if /I "%~1"=="/launch" exit /b 1
pause
exit /b 1

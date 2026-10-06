@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "STUDIO_PYTHON=%~dp0.venv\Scripts\python.exe"
set "STUDIO_PYTHONW=%~dp0.venv\Scripts\pythonw.exe"

if not exist "%STUDIO_PYTHON%" goto refresh_environment
if not exist "%STUDIO_PYTHONW%" goto refresh_environment

rem Refresh an existing environment when requirements.txt has changed.
"%STUDIO_PYTHON%" -c "from pathlib import Path; import hashlib, sys; requirements = Path(r'%~dp0requirements.txt'); stamp = Path(r'%~dp0.venv\requirements.sha256'); expected = hashlib.sha256(requirements.read_bytes()).hexdigest(); raise SystemExit(0 if stamp.is_file() and stamp.read_text(encoding='ascii').strip() == expected else 1)" >nul 2>&1
if errorlevel 1 goto refresh_environment

rem Catch damaged/incomplete environments before pythonw can hide the error.
"%STUDIO_PYTHON%" -c "import tkinter, customtkinter, numpy, sklearn, scipy, joblib, xgboost, shapely, vtkmodules; from studio.ui import run" >nul 2>&1
if errorlevel 1 goto refresh_environment
goto launch_studio

:refresh_environment
echo Preparing Antenna Surrogate Studio...
call setup_windows.bat /launch
if errorlevel 1 (
    echo.
    echo Setup did not complete. Review the message above and try again.
    pause
    exit /b 1
)

"%STUDIO_PYTHON%" -c "import tkinter, customtkinter, numpy, sklearn, scipy, joblib, xgboost, shapely, vtkmodules; from studio.ui import run" >nul 2>&1
if errorlevel 1 (
    echo.
    echo The Studio still cannot start after setup.
    echo Run run_studio.bat to see the full Python error.
    pause
    exit /b 1
)

:launch_studio
start "" /D "%~dp0" "%STUDIO_PYTHONW%" "%~dp0app.py"
if errorlevel 1 (
    echo.
    echo Windows could not start Antenna Surrogate Studio.
    echo Run run_studio.bat to see the full Python error.
    pause
    exit /b 1
)
exit /b 0

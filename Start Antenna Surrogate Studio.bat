@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "STUDIO_PYTHONW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%STUDIO_PYTHONW%" (
    call setup_windows.bat /launch
    if errorlevel 1 (
        echo.
        echo Setup did not complete. Review the message above and try again.
        pause
        exit /b 1
    )
    set "STUDIO_PYTHONW=%~dp0.venv\Scripts\pythonw.exe"
)

start "" "%STUDIO_PYTHONW%" app.py
exit /b 0

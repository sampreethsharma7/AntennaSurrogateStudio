@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "ANTENNA_STUDIO_BUILD_CHANNEL=development"
set "SNOWBUDDY_DEVELOPMENT_LOG=1"

call "Start Antenna Surrogate Studio.bat"
exit /b %errorlevel%

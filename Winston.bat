@echo off
rem Starts Winston as a floating orb with no console window. Logs go to .runtime\winston.log
cd /d "%~dp0"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONIOENCODING=utf-8
start "" ".venv\Scripts\pythonw.exe" -m winston.main

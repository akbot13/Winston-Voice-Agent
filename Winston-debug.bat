@echo off
rem Same as Winston.bat but keeps a console open showing everything Winston hears and says.
title Winston (debug)
cd /d "%~dp0"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -m winston.main
pause

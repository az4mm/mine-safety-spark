@echo off
cd /d "%~dp0"
where java >nul 2>nul
if errorlevel 1 (
  echo Install Java 17 or 21 first, then reopen this window.
  pause
  exit /b 1
)
py -3 -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"
if errorlevel 1 goto fail
echo Setup complete. Run run_demo_windows.bat.
pause
exit /b 0
:fail
echo Setup failed. Read the error above. Python 3.10-3.13 is required.
pause
exit /b 1

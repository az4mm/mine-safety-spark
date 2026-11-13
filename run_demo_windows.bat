@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m mine_safety generate
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m mine_safety batch
if errorlevel 1 goto fail
start "" http://127.0.0.1:8000
.venv\Scripts\python.exe -m mine_safety serve
goto end
:fail
echo The demo failed. Keep this window open and review the error above.
pause
:end

@echo off
setlocal

if not exist ".deps_installed" (
  python -m pip install -r requirements.txt
  if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    exit /b %errorlevel%
  )
  echo Dependencies installed on %date% %time%> .deps_installed
)
python soniox_gui.py

@echo off
setlocal

cd /d C:\Users\sheshou
if %errorlevel% neq 0 (
  echo Failed to switch to C:\Users\sheshou
  pause
  exit /b %errorlevel%
)

if not exist ".deps_installed" (
  python -m pip install -r requirements.txt
  if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b %errorlevel%
  )
  echo Dependencies installed on %date% %time%> .deps_installed
)

python soniox_gui.py
if %errorlevel% neq 0 (
  echo soniox_gui.py exited with error %errorlevel%.
  pause
  exit /b %errorlevel%
)

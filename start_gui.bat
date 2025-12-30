@echo off
setlocal

pushd "%~dp0"

python gui.py
if errorlevel 1 (
  echo.
  echo Run failed. Please ensure Python is installed and available as "python".
)

popd
pause

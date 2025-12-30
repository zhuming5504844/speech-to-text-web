@echo off
setlocal

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

python gui.py
if errorlevel 1 (
  echo.
  echo 运行失败。请确认已安装 Python 并可在命令行中使用 python。
)

pause

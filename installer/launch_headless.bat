@echo off
chcp 65001 >nul
title EyeQ — 实时命令行监测

set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%python_embed\python.exe
set APP_DIR=%SCRIPT_DIR%..

if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到 Python，请先运行 setup_windows.bat 进行安装。
    pause & exit /b 1
)

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║         EyeQ 命令行实时监测模式                     ║
echo ║         EyeQ Headless Real-time Monitor              ║
echo ╚══════════════════════════════════════════════════════╝
echo.
echo   确保摄像头已连接并授权，按 Ctrl+C 停止。
echo   Make sure your camera is connected and permitted.
echo.

cd /d "%APP_DIR%"
"%PYTHON_EXE%" main.py --mode headless
echo.
pause

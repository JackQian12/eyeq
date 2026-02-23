@echo off
chcp 65001 >nul
title EyeQ — 仪表盘 Dashboard

set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%python_embed\python.exe
set APP_DIR=%SCRIPT_DIR%..
set PORT=8501

REM ── 检查 Python 是否安装 ─────────────────────────────────────────
if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到 Python，请先运行 setup_windows.bat 进行安装。
    pause & exit /b 1
)

REM ── 杀掉占用端口的旧进程 ─────────────────────────────────────────
for /f "tokens=5" %%a in ('netstat -aon ^| find ":%PORT% "') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║         EyeQ 可视化仪表盘正在启动...                ║
echo ║         EyeQ Dashboard is starting...                ║
echo ╚══════════════════════════════════════════════════════╝
echo.
echo   稍后浏览器将自动打开 http://localhost:%PORT%
echo   按 Ctrl+C 停止服务
echo.

cd /d "%APP_DIR%"
"%PYTHON_EXE%" -m streamlit run dashboard\app.py ^
    --server.port %PORT% ^
    --server.headless false ^
    --browser.gatherUsageStats false ^
    --theme.base light

@echo off
chcp 65001 >nul
title EyeQ — 一键安装 / One-Click Setup

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║          EyeQ 智能眼镜监测系统 — 安装程序           ║
echo ║      EyeQ Smart Glasses Monitor — Installer          ║
echo ╚══════════════════════════════════════════════════════╝
echo.

REM ── 检测系统架构 ─────────────────────────────────────────────────
set ARCH=amd64
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" set ARCH=arm64
if "%PROCESSOR_ARCHITEW6432%"=="ARM64" set ARCH=arm64

REM ── 路径设置 ─────────────────────────────────────────────────────
set INSTALL_DIR=%~dp0
set PYTHON_DIR=%INSTALL_DIR%python_embed
set PYTHON_EXE=%PYTHON_DIR%\python.exe
set PY_VERSION=3.12.8

echo [1/5] 检查 Python 嵌入式环境...
if exist "%PYTHON_EXE%" (
    echo       Python 已存在，跳过下载。
    goto :install_deps
)

echo [1/5] 下载 Python %PY_VERSION% 嵌入式包 ^(约 10 MB^)...
set PY_ZIP=python-%PY_VERSION%-embed-%ARCH%.zip
set PY_URL=https://www.python.org/ftp/python/%PY_VERSION%/%PY_ZIP%

powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = 'Tls12'; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%INSTALL_DIR%%PY_ZIP%' -UseBasicParsing }"
if errorlevel 1 (
    echo [错误] Python 下载失败，请检查网络连接。
    echo        尝试手动下载：%PY_URL%
    pause & exit /b 1
)

echo       解压中...
mkdir "%PYTHON_DIR%" 2>nul
powershell -Command "Expand-Archive -Path '%INSTALL_DIR%%PY_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
del "%INSTALL_DIR%%PY_ZIP%"

REM ── 启用 site-packages（嵌入式 Python 默认禁用）──────────────────
echo       配置 Python 路径...
for %%f in ("%PYTHON_DIR%\python3*._pth") do (
    powershell -Command "(Get-Content '%%f') -replace '#import site','import site' | Set-Content '%%f'"
)

REM ── 下载 pip ─────────────────────────────────────────────────────
echo [2/5] 安装 pip...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = 'Tls12'; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%INSTALL_DIR%get-pip.py' -UseBasicParsing }"
"%PYTHON_EXE%" "%INSTALL_DIR%get-pip.py" --no-warn-script-location -q
del "%INSTALL_DIR%get-pip.py"

:install_deps
echo [3/5] 安装依赖包（首次约需 3–5 分钟，请耐心等待）...
echo       正在安装 opencv-python...
"%PYTHON_EXE%" -m pip install opencv-python==4.10.0.84 --no-warn-script-location -q
echo       正在安装 mediapipe...
"%PYTHON_EXE%" -m pip install mediapipe==0.10.14 --no-warn-script-location -q
echo       正在安装 streamlit 及其他依赖...
"%PYTHON_EXE%" -m pip install streamlit>=1.32.0 plotly>=5.18.0 pandas>=2.0.0 sqlalchemy>=2.0.0 pyyaml>=6.0 rich>=13.0.0 scipy>=1.11.0 numpy>=1.24.0 --no-warn-script-location -q

echo [4/5] 创建数据目录...
mkdir "%INSTALL_DIR%..\data" 2>nul

echo [5/5] 创建桌面快捷方式...
set DESKTOP=%USERPROFILE%\Desktop
powershell -Command "$s=(New-Object -COM WScript.Shell).CreateShortcut('%DESKTOP%\EyeQ 仪表盘.lnk'); $s.TargetPath='%INSTALL_DIR%launch_dashboard.bat'; $s.WorkingDirectory='%INSTALL_DIR%'; $s.IconLocation='%SystemRoot%\System32\shell32.dll,23'; $s.Description='EyeQ 眼部健康监测仪表盘'; $s.Save()"
powershell -Command "$s=(New-Object -COM WScript.Shell).CreateShortcut('%DESKTOP%\EyeQ 命令行监测.lnk'); $s.TargetPath='%INSTALL_DIR%launch_headless.bat'; $s.WorkingDirectory='%INSTALL_DIR%'; $s.IconLocation='%SystemRoot%\System32\shell32.dll,20'; $s.Description='EyeQ 命令行实时监测'; $s.Save()"

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║   ✓ 安装完成！桌面已创建快捷方式。                  ║
echo ║   Installation complete! Desktop shortcuts created.  ║
echo ╚══════════════════════════════════════════════════════╝
echo.
echo   使用方式 / Usage:
echo   · 双击桌面"EyeQ 仪表盘"  → 浏览器可视化界面
echo   · 双击桌面"EyeQ 命令行"   → 终端实时指标输出
echo.
pause

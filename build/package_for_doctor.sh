#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# package_for_doctor.sh
# 在 macOS/Linux 上运行，将项目打包为可发给 Windows 用户的 zip 文件。
# 对方仅需运行 setup_windows.bat 即可一键安装（无需预装 Python）。
#
# 用法 / Usage:
#   chmod +x build/package_for_doctor.sh
#   ./build/package_for_doctor.sh
#
# 产物：EyeQ_Windows_portable_<日期>.zip
# ─────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
DATE=$(date +%Y%m%d)
OUT_ZIP="$ROOT/EyeQ_Windows_portable_${DATE}.zip"

echo "╔══════════════════════════════════════════════════════╗"
echo "║   EyeQ Windows 便携包打包脚本                       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 临时目录 ─────────────────────────────────────────────────────────────────
TEMP_DIR=$(mktemp -d)
PACKAGE_DIR="$TEMP_DIR/EyeQ_Windows"
mkdir -p "$PACKAGE_DIR"

echo "[1/5] 复制核心源代码..."
# 复制 Python 源码模块
for dir in vision metrics dashboard; do
    cp -r "$ROOT/$dir" "$PACKAGE_DIR/"
done
cp "$ROOT/main.py"          "$PACKAGE_DIR/"
cp "$ROOT/requirements.txt" "$PACKAGE_DIR/"
cp "$ROOT/config.yaml"      "$PACKAGE_DIR/"

echo "[2/5] 复制安装器和启动脚本..."
cp "$ROOT/installer/setup_windows.bat"       "$PACKAGE_DIR/"
cp "$ROOT/installer/launch_dashboard.bat"    "$PACKAGE_DIR/"
cp "$ROOT/installer/launch_headless.bat"     "$PACKAGE_DIR/"
cp "$ROOT/installer/INSTALL_医生版使用说明.txt" "$PACKAGE_DIR/使用说明.txt"

echo "[3/5] 创建数据目录..."
mkdir -p "$PACKAGE_DIR/data"
touch "$PACKAGE_DIR/data/.gitkeep"

echo "[4/5] 清理不必要文件（__pycache__、.DS_Store 等）..."
find "$PACKAGE_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$PACKAGE_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$PACKAGE_DIR" -name ".DS_Store" -delete 2>/dev/null || true
find "$PACKAGE_DIR" -name "*.db" -delete 2>/dev/null || true

echo "[5/5] 生成 ZIP 包..."
cd "$TEMP_DIR"
zip -r "$OUT_ZIP" "EyeQ_Windows/" -x "*/\.*" > /dev/null

# ── 清理临时目录 ─────────────────────────────────────────────────────────────
rm -rf "$TEMP_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✓ 打包完成！                                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  输出文件：$OUT_ZIP"
echo "  文件大小：$(du -sh "$OUT_ZIP" | cut -f1)"
echo ""
echo "  发给医生的使用方法："
echo "  1. 将 zip 发送给医生（邮件/微信/U盘均可）"
echo "  2. 医生解压后，双击 setup_windows.bat"
echo "  3. 等待自动安装完成（约3-5分钟）"
echo "  4. 桌面出现快捷方式后即可使用"
echo ""

# -*- mode: python ; coding: utf-8 -*-
# ─────────────────────────────────────────────────────────────────────────────
# eyeq_windows.spec
# PyInstaller 打包配置 — 生成 Windows 单文件夹可执行程序
#
# 在 Windows 上运行：
#   pip install pyinstaller
#   pyinstaller build/eyeq_windows.spec
#
# 产物在 dist/EyeQ/ 目录，打包 zip 发给医生即可。
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
import mediapipe
import streamlit

ROOT = Path(SPECPATH).parent          # 项目根目录
MP_DIR  = Path(mediapipe.__file__).parent
ST_DIR  = Path(streamlit.__file__).parent

block_cipher = None

# ── 需要额外收集的数据文件 ────────────────────────────────────────────────────
added_datas = [
    # MediaPipe 模型文件（必须）
    (str(MP_DIR / "modules"),     "mediapipe/modules"),
    (str(MP_DIR / "python" / "solutions"), "mediapipe/python/solutions"),
    # Streamlit 静态资源
    (str(ST_DIR / "static"),      "streamlit/static"),
    (str(ST_DIR / "runtime"),     "streamlit/runtime"),
    # 项目配置和仪表盘
    (str(ROOT / "config.yaml"),   "."),
    (str(ROOT / "dashboard"),     "dashboard"),
]

# ── 隐式导入（PyInstaller 无法自动发现）─────────────────────────────────────
hidden_imports = [
    "mediapipe",
    "mediapipe.python.solutions.face_mesh",
    "mediapipe.python.solutions.drawing_utils",
    "cv2",
    "streamlit",
    "streamlit.web.cli",
    "streamlit.runtime.scriptrunner.magic_funcs",
    "sqlalchemy.dialects.sqlite",
    "scipy.special._cdflib",
    "plotly",
    "vision.eye_tracker",
    "vision.blink_detector",
    "vision.tear_film",
    "metrics.aggregator",
    "metrics.storage",
    "altair",
    "pydeck",
    "pyarrow",
]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=added_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PyQt5", "PySide2", "wx", "IPython"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # 使用 COLLECT 模式（单文件夹，更稳定）
    name="EyeQ",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                      # 不使用 UPX 压缩（避免杀毒误报）
    console=True,                   # headless 模式需要控制台
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="EyeQ",
)

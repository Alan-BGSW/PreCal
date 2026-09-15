# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [
    ('app.py', '.'),
    ('pipeline.py', '.'),
    ('a2l_hex_to_cdfx.py', '.'),
    ('cdfx_to_a2l_hex.py', '.'),
    ('apply_frm_start_values.py', '.'),
    ('extract_frm.py', '.'),
    ('detailed_process_logs.py', '.'),
    ('log_delivery.py', '.'),
    ('parser.py', '.'),
    ('.streamlit', '.streamlit'),
]
binaries = []
hiddenimports = []
datas += copy_metadata('streamlit')
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# extract_frm.py imports these lazily inside functions; include them explicitly.
hiddenimports += [
    'pdfplumber',
    'fitz',
    'pytesseract',
    'PIL',
]

for pkg_name in ('pdfplumber', 'pytesseract', 'PIL'):
    pkg_ret = collect_all(pkg_name)
    datas += pkg_ret[0]
    binaries += pkg_ret[1]
    hiddenimports += pkg_ret[2]


a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CalAiTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CalAiTool',
)

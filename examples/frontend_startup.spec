# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['MIW_autofit.py'],
    pathex=[],
    binaries=[],
    datas=[('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/icon.ico','.'),
           ('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/splash.png','.'),
           ('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/plots','plots'),
           ('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/data','data')],
    hiddenimports=['autofit'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    exclude_binaries=True,
    name='frontend_startup',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='frontend_startuo',
)

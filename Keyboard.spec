# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

pygame_data = collect_all('pygame')

a = Analysis(
    ['Keyboard.py'],
    pathex=[],
    binaries=pygame_data[1],
    datas=pygame_data[0],
    hiddenimports=pygame_data[2],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='Keyboard',
    debug=False,
    strip=False,
    upx=True,
    console=False,
    onefile=True,
)

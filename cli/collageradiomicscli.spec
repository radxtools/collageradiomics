# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(['collageradiomicscli.py'],
             pathex=[os.getcwd()],
             binaries=[],
             datas=[],
             hiddenimports=['skimage', 'skimage.io', 'skimage.transform', 'skimage.filter.rank.core_cy', 'skimage._shared.interpolation', 'skimage.feature', 'skimage.feature._orb_descriptor_positions'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='collageradiomics',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )

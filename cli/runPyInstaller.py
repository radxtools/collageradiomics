import os
import PyInstaller
from shutil import copyfile

print("Creating exe")

os.system("pyinstaller --onefile collageradiomicscli.spec")

print("All Done")
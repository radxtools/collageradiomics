## Instructions to compile CLI

```powershell
conda create -p ./venv python=3.6.5 -y # use 3.7 for macOS, because...macOS
conda activate ./venv
pip install -e .
conda install -c conda-forge scikit-image -y
python ./runPyInstaller.py
```

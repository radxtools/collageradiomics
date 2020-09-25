## Instructions to compile CLI

```powershell
conda create -p ./venv python=3.6.5 -y
conda activate ./venv
pip install -e .
conda install -c conda-forge scikit-image
python ./runPyInstaller.py
```

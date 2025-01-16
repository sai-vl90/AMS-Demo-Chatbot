python -m pip install --upgrade pip setuptools wheel
pip uninstall -y azure-core azure-identity
pip install --upgrade --force-reinstall -r requirements.txt

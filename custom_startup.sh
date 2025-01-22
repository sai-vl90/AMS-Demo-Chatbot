#!/bin/bash
apt-get update
apt-get install  -y  libgl1-mesa-glx  libglib2.0-0
pip install  -r  requirements.txt

# 1) Remove /agents/python from PYTHONPATH
#    or simply replace the entire PYTHONPATH to point only to your venv

# Approach A: Try to surgically remove /agents/python
export PYTHONPATH=$(echo "$PYTHONPATH" | sed 's/\/agents\/python://')

# Approach B: Or just forcibly set PYTHONPATH to your venv site-packages:
# export PYTHONPATH=/home/site/wwwroot/antenv/lib/python3.12/site-packages


# 2) Launch your app with Waitress
python startup.py

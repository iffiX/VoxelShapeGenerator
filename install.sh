#!/bin/bash
python3 -m venv venv
venv/bin/pip install wheel
venv/bin/pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
venv/bin/pip install git+https://github.com/openai/shap-e.git
venv/bin/pip install fvcore iopath trimesh PyQt5==5.15.2
VERSION_STR=`venv/bin/python version_str.py`
venv/bin/pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/$VERSION_STR/download.html
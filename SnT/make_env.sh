#!/usr/bin/env bash

set -x
set -e

# Prompt the user to enter an installation directory or use the current directory as default
read -p "Enter the installation directory (default is current directory): " INSTALL_DIR
INSTALL_DIR=${INSTALL_DIR:-${PWD}}

# Create temporary directory if it exists, then set TMPDIR for package installations
if [ -d "${INSTALL_DIR}/tmp" ]; then 
  rm -r "${INSTALL_DIR}/tmp"
fi

mkdir -p "${INSTALL_DIR}/tmp"
export TMPDIR="${INSTALL_DIR}/tmp" # in case the default tmp dir is not big enough to install pytorch
export PIP_CACHE_DIR="${INSTALL_DIR}/tmp/.cache" # Redirect pip cache to current working directory

# Create and activate virtual environment in specified directory
python3 -m venv "${INSTALL_DIR}/env"
source "${INSTALL_DIR}/env/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install tqdm
pip install matplotlib
pip install mplhep
pip install numba
pip install numpy
pip install pandas
pip install pyarrow
pip install scikit-learn
pip install scipy
pip install tabulate
pip install tensorboard
pip install torch
pip install torchviz
pip install uproot
pip install xgboost==1.2.1
pip install pyyaml
pip install dill

# Clean up temporary directory
rm -r "${INSTALL_DIR}/tmp"

# LearnControllers

Code for NeurIPS 2019

# Installation
## Install Miniconda (or Anaconda)
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Take note of where the package is installed (likely /home/<username>/miniconda3). Run
```
/home/<username>/miniconda3/bin/conda init
conda config --set auto_activate_base false
```
to initialize and disable the automatic activation of the base environment.

## Create a conda environment
```
# To call the environment venv:
conda create --name venv python=3.7
```
## Install dependencies
```
# Activate conda environment if it isn't
conda activate venv
# Navigate to wherever you cloned this repo
pip install -r requirements.txt
# (I personally had to do pip install -r --no-cache-dir requirements.txt because of a memory error)
# Setup the python wrappers for rvo2, used for simulation.
cd learn_failure/pyrvo2
# Install Cython. This did not work with pip for me
conda install Cython
# If you don't have cmake installed, install it now (depends on distro)

```

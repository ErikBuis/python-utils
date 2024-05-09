# Mamba
[Mamba](https://mamba.readthedocs.io/en/latest/) is a (much) faster and smaller drop-in replacement for the popular [Conda](https://docs.conda.io/en/latest/) package manager. We recommend using Mamba to install the required Python modules, since it uses the exact same command syntax as Conda with the added benefit of being more efficient.

# Installation
The guide at [this link](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) describes how to install Mamba, but for your convenience we have put the necessary commands below. Start by running the following commands in your terminal:
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
Then, restart your shell. We recommend entering `conda config --set auto_activate_base false` (yes, start the command with `conda`) if you do not want Conda/Mamba to activate the base environment every time you start a terminal window.

# Usage
Mamba commands do not differ in a significant way from their Conda equivalents. Below are a few examples of frequently used commands:
- To create an environment from a `environment.yml` file, run:
```bash
mamba env create -f environment.yml
```
- To activate an environment, run:
```bash
mamba activate your-env-name
```
- To deactivate an environment, run:
```bash
mamba deactivate
```
- To install a package in an environment, run:
```bash
mamba install -n your-env-name package-name
```
- To install a package in an environment with an `environment.yml` file, first add the package you want to install to the file and then run:
```bash
mamba env update -f environment.yml --prune
```
- To remove a package from an environment, run:
```bash
mamba remove -n your-env-name package-name
```
- To remove an environment, run:
```bash
mamba remove -n your-env-name --all
```
- To list all environments, run:
```bash
mamba env list
```

## Template `environment.yml` file
For your convenience, we have provided a template `environment.yml` file below. This file contains the some basic packages most data science projects will depend on. You can copy this file to your repository and edit it to create your own environment with Mamba or Conda.
```yaml
name: your-env-name
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  # General
  - python=3.10
  - ipython
  - ipykernel
  - tqdm
  - requests
  - pytest
  # PyTorch
  - pytorch=2.1.0
  - torchvision=0.16.0
  - torchaudio=2.1.0
  - pytorch-cuda=12.1
  - lightning=2.1.3
  - tensorboard
  # Scientific
  - numpy
  - scipy
  - scikit-learn
  - matplotlib
  - pillow
  # Tabular data
  - pandas
  - pyarrow
```

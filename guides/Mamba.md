# Mamba
[Mamba](https://mamba.readthedocs.io/en/latest/) is a (much) faster and smaller drop-in replacement for the popular [Conda](https://docs.conda.io/en/latest/) package manager. We recommend using Mamba to install the required Python modules, since it uses the exact same command syntax as Conda with the added benefit of being more efficient. Note that while you can still use `conda` commands after installing Mamba, you should opt for using the `mamba` variants.


# Installation
The guide at [this link](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) describes how to ***install Mamba***, but for your convenience we have put the necessary commands below.

## Linux/Unix
For a Linux/Unix-like OS, start by running the following commands in your terminal:
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
Then, ***restart your shell***. We recommend entering `conda config --set auto_activate_base false` (yes, start the command with `conda`) if you do not want Conda/Mamba to activate the base environment every time you start a terminal window.

## Windows
For Windows, first download the installer at [this link](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe). When running the installation wizard, check the box of the option "Add Miniforge3 to my PATH environment variable". Next, start a PowerShell window and type the following commands:
```powershell
conda init powershell
```
Now ***restart PowerShell***. If you find some error like this in PowerShell,
> \WindowsPowerShell\profile.ps1 cannot be loaded because running scripts is disabled on this system. For more information, see about_Execution_Policies at <https:/go.microsoft.com/fwlink/?LinkID=135170>. At line:1 char:3
then you should change the execution policy using:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Finally, ***restart PowerShell*** one last time. We recommend entering `conda config --set auto_activate_base false` (yes, start the command with `conda`) if you do not want Mamba/Conda to activate the base environment every time you start a terminal window. Note that the `mamba activate your-env-name` and `mamba deactivate` commands do not work on Windows, thus you should use `conda` instead for these specific cases.


# Usage
For both Conda and Mamba environments, we strongly recommend to only install packages with `mamba install` or `conda install`, never with `pip install`! This is because when packages are installed using Pip, Mamba/Conda won't track the installed packages which makes future `mamba install` or `conda install` commands malfunction (more precisely, multiple versions of a single package may be installed at the same time, a package can be overwritten or it can lose its dependencies. In other words, your environment will become unusable after this and you will need to install the whole environment from scratch again). Thus, always check if there is a way to install your package using Mamba/Conda first!

Since Mamba is designed to be a drop-in replacement for Conda, the command syntax is the same in almost all common use cases. Below are a few examples of frequently used commands:
- To create a blank environment with only Python 3.x installed, run `mamba create -n your-env-name python=3.x`.
- To create an environment from an `environment.yml` file, run `mamba env create -f environment.yml`.
- To activate an environment, run `mamba activate your-env-name` (or `conda activate your-env-name` on Windows).
- To deactivate an environment, run `mamba deactivate` (or `conda deactivate` on Windows).
- To install a package in an environment without an `environment.yml` file, run `mamba install -n your-env-name package-name`.
- To install a package in an environment with an `environment.yml` file, first add the package you want to install to the file and then run `mamba env update -f environment.yml --prune`.
- To remove a package from an environment, run `mamba remove -n your-env-name package-name`.
- To remove an environment, run `mamba remove -n your-env-name --all`.
- To list all environments, run `mamba env list`.
- To list all packages in the current environment, run `mamba list`.


# Installing a git repository as a Python package
To install a git repository as a Python package, go to the repository's page on github.com and copy the `Clone > SSH` command. This will look something like `git@github.com:user-name/repository-name.git`. Unfortunately, it is not entirely straightforward to just install using something like `mamba install`. Instead, we have to install with pip. The following steps are necessary to get the correct Pip command.
1. Substitute the colon `:` by a slash `/`. For example, your URI will now look like: `git@github.com/user-name/repository-name.git`
2. Add `git+ssh://` to the start of the URL. For example, your URI will now look like: `git+ssh://git@github.com/user-name/repository-name.git`
3. Search the repository's root folder for an installation configuration file.
   - If there is a `pyproject.toml` file, look for a `name = "<name>"` entry under the `[project]` section. Copy the `<name>` field.
   - If there is a `setup.cfg` file, look for a `name = <name>` entry under the `[metadata]` section. Copy the `<name>` field.
4. Add `#egg=<name>` to the end of the URL. For example, if the `<name>` field you found in step 3 was `myrepositoryname`, your URI will now look like: `git+ssh://git@github.com/user-name/repository-name.git#egg=myrepositoryname`
5. Install the repository into your project by running `pip install --editable git+ssh://git@github.com/user-name/repository-name.git#egg=myrepositoryname` or adding the following to your environment.yml file:
```yaml
  - pip:
    - --editable git+ssh://git@github.com/user-name/repository-name.git#egg=myrepositoryname
```
This will create a folder named `src` into your root directory, which will have all repositories you installed using this method in it. The names of the repositories will be the same as the ones you specified in the `#egg=<name>` field. For example, your directory structure will now look like:
<pre>
.
├── … <b>file.extension</b>: Any file in your own project.
├── …
└── 📁 <b>src</b>
    └── 📁 <b>myrepositoryname</b>: Code for the repository at https://github.com/user-name/repository-name.git
</pre>


# Template `environment.yml` file
For your convenience, we have provided a template `environment.yml` file below. This file contains the a lot of packages data science projects may depend on, so feel free to remove any packages you don't need for your project. You can copy this file to your repository and edit it to create your own environment with Mamba or Conda.
```yaml
name: your-env-name

channels:
  - pytorch
  - nvidia
  - conda-forge

dependencies:
  # Base Python
  - python=3.10
  - pip

  # Development tools
  - pre-commit
  - ipython
  - ipykernel
  - tqdm
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
  - scikit-image
  - matplotlib
  - pillow
  - opencv

  # Tabular data
  - pandas
  - pyarrow

  # Geospatial data
  - geopandas
  - shapely
  - pyogrio
  - fiona
  - rasterio
  - rio-cogeo
  # - gdal (if you include this, you must specify numpy<2)

  - pip:
    # Install packages that aren't available on conda-forge
    # - package-not-available-via-conda-forge

    # Install private packages (will be installed in `./src/*`)
    # - --editable git+ssh://git@github.com/user-name/repository-name.git#egg=myrepositoryname
```


# Extras
The following parts of the tutorial are totally optional and only for those who want to further customize their Mamba environment. If you are not interested in this, you are now done with the installation process. Enjoy the Mamba package manager!

## IPython autoreload
When testing stuff in IPython, it can be useful to have IPython automatically reload modules when they change. To enable this, you should edit your config file. First, generate a default config file by running:
```bash
ipython profile create
```
Then, open the file `~/.ipython/profile_default/ipython_config.py` and change the line
```python
# c.InteractiveShellApp.extensions = []
```
to
```python
c.InteractiveShellApp.extensions = ['autoreload']
```
and the line
```python
# c.InteractiveShellApp.exec_lines = []
```
to
```python
c.InteractiveShellApp.exec_lines = ['%autoreload 2']
```

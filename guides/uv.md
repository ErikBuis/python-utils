# uv guide
[uv](https://github.com/astral-sh/uv) is a package installer for Python, designed to be significantly faster and more efficient than `pip` or `conda`/`mamba` while maintaining compatibility with existing workflows. We recommend using `uv` to install Python packages wherever possible due to its superior performance and enhanced dependency resolution capabilities.

# Installation
The official installation guide is available [here](https://github.com/astral-sh/uv#installation), but for convenience, the necessary commands are provided below.

## Windows
Run the following command to install `uv`:
```powershell
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Linux/macOS
Run the following command to install `uv`:
```bash
# On Linux/macOS.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Verifying the Installation
To verify that `uv` is installed correctly, run `uv` in the terminal. This will print the help page.

## Updating `uv`
To update `uv` to the latest version, run the following command:
```powershell
uv self update
```

## Uninstalling `uv`
To uninstall `uv`, run the following commands:
1. Clean up stored data (optional):
    ```powershell
    uv cache clean
    rm -r "$(uv python dir)"
    rm -r "$(uv tool dir)"
    ```
2. Remove the `uv` and `uvx` binaries:\
    Windows:
    ```powershell
    $ rm $HOME\.local\bin\uv.exe
    $ rm $HOME\.local\bin\uvx.exe
    ```
    Linux/MacOS:
    ```bash
    rm ~/.local/bin/uv ~/.local/bin/uvx
    ```


# Usage

## Project-based Workflow
`uv` is designed to work in a workspace-based manner. This means that it will create a `.venv` in the current workspace directory where it will store all the packages you install. This is done to avoid conflicts between different projects. For creating a global environment (discouraged), please see the [section below](#use-a-global-environment). The rest of this tutorial assumes that you are working from within a workspace.
<br><br>

It is *strongly discouraged* to modify the project environment using the `pip` interface, e.g. with `uv pip install`. This will cause dependency conflicts, since `uv` can not manage `pip`-managed packages. Instead, use `uv add` to add packages to the environment (see the [section on installing, updating and removing packages](#installing-updating-and-removing-packages) below). For one-off requirements, use `uv run --with` (see the [section on using the environment](#using-the-environment) below).

## The `uv.lock` File
`uv` creates a `uv.lock` file next to the `pyproject.toml`.
<br><br>

Unlike the `pyproject.toml`, which is used to specify the broad requirements of your project, the lockfile contains the exact resolved versions that are installed in the project environment. This file should be checked into `git` version control, allowing for consistent and reproducible installations across machines.
<br><br>

`uv.lock` is a human-readable TOML file but is managed by `uv` and should not be edited manually. There is no Python standard for lockfiles at this time, so the format of this file is specific to `uv` and not usable by other tools.

## Commands
This subsection provides a brief overview of the most commonly used `uv` commands. For a complete list of commands, refer to the [official documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/).

### Create a new environment
- ***To sync/install an environment that is defined in a `pyproject.toml` file located in the current working directory (e.g. created by a colleague), use:***
    ```powershell
    uv sync
    ```
- `uv` uses a `pyproject.toml` file to keep track of the environment's dependencies. To initialize a `uv` environment in the current working directory, use the following command. If you already have a `pyproject.toml` file, you can skip this step.
    ```powershell
    uv init --bare --python '>=3.11, <3.12'
    ```
    Please replace `example` with the name of your project, and `'>=3.11, <3.12'` with the desired Python version.<br>
    To change the Python version of an existing project, change the `python` field in the `pyproject.toml` file and run `uv sync`.

### Using the environment
- We recommend activating your `uv` environment in the terminal, so that you don't have to specify the environment in all subsequent commands. To activate the `uv` environment, run the following command:<br>
    Windows:
    ```powershell
    .venv\Scripts\activate
    ```
    Linux/macOS:
    ```bash
    source .venv/bin/activate
    ```
- To deactivate the `uv` environment, run the following command:
    ```powershell
    deactivate
    ```
- If your environment is *not* activated but your workspace does contain a `pyproject.toml`, and you just want to quickly run a Python file in your workspace, you can use the `uv run` command.
    - To run a script in the current workspace's virtual environment:
        ```powershell
        uv run my_script.py
        ```
    - If you want to just quickly run a script with a specific (extra) dependency (with or without a specific version) without adding it to your `pyproject.toml`, you can use the `--with` flag:
        ```powershell
        uv run --with [package-name] my_script.py
        ```
        For multiple dependencies, repeat the `--with` flag for each one.
    - To specify a different Python version to use temporarily, you can use the `--python` flag:
        ```powershell
        uv run --python 3.10 my_script.py
        ```
        Replace `3.10` with your desired Python version.

### Installing, updating and removing packages
- You can install packages from different sources:
    | Type | Template | Example | Comments |
    |------|----------|---------|----------|
    | From PyPI | `uv add [package-name]` | `uv add numpy` | - |
    | From an index | `uv add [package-name] --index [index-url]` | <code>uv add torch --index pytorch-cu124=https:<wbr>//download.pytorch.org<wbr>/whl/cu124</code> | - |
    | From a Git repository | `uv add git+[url-to-repo]` | <code>uv add git+https:<wbr>//github.com<wbr>/ErikBuis/python-utils</code> | You can pin the package to a specific revision (`--rev commit-hash`) or branch (`--branch branch-name`). |
    | From a local path | `uv add [path-to-package]` | `uv add ./GDAL-3.4.3-cp311-cp311-win_amd64.whl` | A local path typically ends in `.whl`, `.tar.gz`, or `.zip`. See [here](https://docs.astral.sh/uv/concepts/resolution/#source-distribution) for all supported formats. For editable installs, add `--editable` in front of the path. |
    | From a `requirements.txt` file | `uv add -r requirements.txt` | - | - |
- To remove a package from the current workspace's virtual environment:
    ```powershell
    uv remove [package-name]
    ```
- To upgrade a package, ignoring pinned versions in any existing output file:
    ```powershell
    uv add --upgrade-package [package-name]
    ```
- To specify an optional dependency ("extras") for a package:
    ```powershell
    uv add [package-name] --optional [extra-name]
    ```
    Users can install the optional dependencies of your project by running e.g. `uv add package-name[extra-name]` or `pip install package-name[extra-name]`.
- To specify a development dependency, which are local-only and will *not* be included in the project requirements when published to PyPI or other indexes:
    ```powershell
    uv add --dev [package-name]
    ```
    This option should be used for packages that are only needed during development, such as `pytest`, `ipython` or `black`.
- You can also edit the `pyproject.toml` file directly to add or remove dependencies. After editing the file, you must syncronize the environment with the new dependencies:
    ```powershell
    uv sync
    ```
    By default, `uv` will prefer the locked versions of packages when running `uv sync` and `uv lock` with an existing `uv.lock` file. Package versions will only change if the project's dependency constraints exclude the previous, locked version. In both of the following cases, upgrades are limited to the project's dependency constraints. For example, if the project defines an upper bound for a package then an upgrade will not go beyond that version.
    - To upgrade all packages:
        ```powershell
        uv lock --upgrade
        ```
    - To upgrade a single package to the latest version, while retaining the locked versions of all other packages:
        ```powershell
        uv lock --upgrade-package [package-name]
        ```
- The cache can grow quite a lot and quickly, because `uv` uses it as much as possible. Cleaning up may be a good idea once in a while.
    - To remove all *unused* cache entries:
        ```powershell
        uv cache prune
        ```
        For example, the cache directory may contain entries created in previous `uv` versions that are no longer necessary and can be safely removed. The following command is safe to run periodically, to keep the cache directory clean.
    - To remove *all* cache entries:
        ```powershell
        uv cache clean
        ```

## Use a "global" environment
Using a global environment is not recommended since it undermines the point of virtual environments. What you should do instead is create a new virtual environment for each project. Note that this does not cost much in terms of disk space, as the environment dependencies *are* installed globally by `uv` and only the project-specific dependencies are stored in the `.venv` directory.

However, if you have a use case where you might want the same environment for multiple projects and you really want to use a global environment, you can use the following trick:
1. Create a global environment:
    ```powershell
    uv init C:\Users\username\path_to_env --bare --python '>=3.11, <3.12'
    ```
    Replace `C:\Users\username\path_to_env` with the path where you want to create the global environment.
2. Activate the global environment:
    ```powershell
    C:\Users\username\path_to_env\Scripts\activate
    ```
3. Now you can install packages in this "global" environment and run Python like you would in a project environment:
    ```powershell
    uv add numpy
    python -c "import numpy; print(numpy.__version__)"
    ```
Alternatively, you can run a script on a one-off basis with your "global" environment:
```powershell
uv run --project C:\Users\username\path_to_env my_script.py
```

Please keep in mind that this means each environment must be associated with a specific directory on your computer. If you want multiple "global" environments, you will need to create a new directory for each one.

### Troubleshooting global environments
Due to the unintended nature of global environments, you may run into issues when using them. During my time with `uv`, my colleagues and I have encountered the following issues:
- **Activate the correct environment**: Make sure you are activating the correct environment before running any Python commands. The active environment is shown in front of the terminal prompt.
- **Conflicts with another environment in a parent package**: If you have a virtual environment in your current working directory you'd like to use and have activated, you could run into issues with a "global" environment you might have created in a parent directory. This is because your subfolder will automatically get recognized as a sub-workspace by `uv`. Your parent folder's `pyproject.toml` probably has the following lines:
    ```toml
    [tool.uv.workspace]
    members = ["subfolder"]
    ```
    To avoid this, you can use the `--no-workspace` flag when running `uv init` in the parent directory, or manually remove the `members` line from the `pyproject.toml` file and running `uv sync` in the subfolder.

## Using `uv` with PyCharm
There may be some additional steps required to use `uv` with PyCharm. The following guide should help you set up `uv` in PyCharm: https://www.jetbrains.com/help/pycharm/uv.html

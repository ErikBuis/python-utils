# Windows Subsystem for Linux (WSL) with CUDA
In order to run CUDA on Windows Subsystem for Linux (WSL), we need to install both WSL and CUDA. However, this can be a cumbersome process as installing CUDA at the wrong time or with the wrong command can overwrite the correct CUDA drivers. To help you through the installation process, this document will guide you through the necessary steps.

## Windows Subsystem for Linux (WSL)
Windows Subsystem for Linux (WSL) is a Windows feature that enables users to run native Linux applications, containers and command-line tools directly on the Windows operating system.

## CUDA
CUDA is a parallel computing platform and application programming interface (API) model created by Nvidia. It allows software developers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing.


# Installation
To install WSL and CUDA on Windows, follow the steps below:
1. First download and install the appropriate NVIDIA driver for GPU support from [here](https://www.nvidia.com/Download/index.aspx). You can view your GPU's name by going to the Task Manager app and opening the `Performance > GPU` panel. Note that you must only install this driver and only on Windows. Do not install any Linux drivers after installing WSL, because this will overwrite the WSL NVIDIA driver.
2. Now install WSL. For future reference, this part of the tutorial is based on [this](https://docs.microsoft.com/en-us/windows/wsl/install) webpage (you don't have to look at this page unless the below commands fail to execute). Start by opening the Windows PowerShell and enter:
    ```powershell
    wsl --status
    ```
    If the message displayed does not report any errors, enter:
    ```powershell
    wsl --install --no-distribution
    ```
    Now reboot the system for changes to take effect. After rebooting, we recommend installing the latest version of Ubuntu, which is Ubuntu 22.04 LTS at the time of writing. Of course, feel free to install another distribution from the list of distributions seen in `wsl --list --online`.
    ```powershell
    wsl --install --distribution Ubuntu-22.04
    ```
    Check if it has been installed correctly by running:
    ```powershell
    wsl --list --verbose
    ```
    You should now see something like the following:
    ```
      NAME            STATE           VERSION
    * Ubuntu-22.04    Stopped         2
    ```
    Finally, open a Bash terminal by opening the newly installed WSL app and typing to make the system fully up-to-date.
    ```bash
    sudo apt update && sudo apt upgrade
    ```
3. If we would install the CUDA toolkit via Linux's `apt` command, we would install the Linux NVIDIA drivers which would overwrite the WSL ones. Thus, do NOT install any packages named `cuda`, `cuda-12-x`, or `cuda-drivers` as these packages will overwrite the WSL NVIDIA driver under WSL 2. Instead, we are going to install the `cuda-toolkit-12-x` metapackage only by using [this](https://developer.nvidia.com/cuda-downloads) download chooser utility. On the webpage, select `Linux > x86_64 > WSL-Ubuntu > 2.0 > deb (local)`. The website then suggests a sequence of commands to execute, which at the time of writing are:
    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb
    sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb
    sudo cp /var/cuda-repo-wsl-ubuntu-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-3
    ```
    After this, you can check if the installation was successful by running `nvidia-smi` in the WSL terminal. If you see a table with information about your GPU, the installation was successful.
4. To prevent a rare bug with Cuda and get rid of the message `Error: libcuda.so: cannot open shared object file: No such file or directory` that you may have seen pop up a few times during the installation process, add the following lines to your `~/.bashrc` file:
    ```bash
    # Fix for CUDA 'libcuda.so: cannot open shared object file' error.
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
    ```
    Then, restart the WSL terminal and enter `echo $LD_LIBRARY_PATH` to check if the variable has been set correctly.
5. In order to be able to plot figures using matplotlib within WSL, you need to install an X server on Windows. We recommend using [VcXSrv](https://sourceforge.net/projects/vcxsrv/). After installing, open the program and make sure that the "Disable access control" box is checked. Then, add the following lines to your `~/.bashrc` file:
    ```bash
    # Suppress warning from vcxsrv.exe in WSL.
    export XDG_RUNTIME_DIR=/tmp/vcxsrv
    ```
    Finally, restart the WSL terminal and enter `echo $XDG_RUNTIME_DIR` to check if the variable has been set correctly.


# Extras
The following parts of the tutorial are totally optional and only for those who want to further customize their WSL environment. If you are not interested in this, you are now done with the installation process. Enjoy your CUDA-enabled WSL environment!


## Install Remote Development extension pack in VSCode
If you are using Visual Studio Code and WSL, you can install the ["Remote Development"](https://aka.ms/vscode-remote/download/extension) extension pack to be able to open folders in the WSL file system.

## Better command prompt
By default, the command prompt in Linux includes the full path to the current working directory (cwd). Sometimes this may become too long and annoying. To make the command prompt a bit more readable and only show the name of the cwd, replace "\w" by "\W" in the following lines in `~/.bashrc`.
```bash
if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
```

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

# Git
Git is a version control system that is widely used in the software development industry. This file will guide you through the installation and configuration of various parts of the GitHub implementation of git.

# Install GitHub CLI
We recommend ***installing GitHub CLI*** so that further operations will be easier. [This](https://github.com/cli/cli?tab=readme-ov-file#installation) guide describes this process, but for convenience we have put the necessary commands for a Linux or WSL install below:
```bash
type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
```

# Set up an SSH key
If you want to clone a private GitHub repository to your local machine, you first have to make GitHub recognize and trust your PC as an authorized entity. To do this, you have to ***generate an SSH keypair and add to to GitHub***. To check if you have already created an SSH key previously, enter `ls -al ~/.ssh`. If you see the entry `id_ed25519.pub`, you can skip this step. Otherwise, please follow one of the following subsections:
- If you installed GitHub CLI in the last step, please follow the section `Set up an SSH key using GitHub CLI`.
- If you did not set up GitHub CLI, you should follow `Set up an SSH key using Bash`.

## Set up an SSH key using GitHub CLI
If you have installed GitHub CLI, you can set up an SSH key by running the following command in Bash:
```bash
gh auth login
```
Choose `GitHub.com` as the account you want to log in to, then select `SSH` as the preferred protocol and select `Yes` when asked to generate a new SSH key to add to you GitHub account. Finally, give the key a descriptive name like `Your-pc-name WSL SSH Key`.

## Set up an SSH key using Bash
[This](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) guide describes how to set up an SSH keypair manually, but to save some time we have summarized the commands to enter below. Enter the following commands to create a new key:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```
Next, go to your GitHub account, look for `Settings > SSH and GPG keys > New SSH key`, enter a descriptive name like `Your-pc-name WSL SSH key` and finally copy-paste the public key (retrievable via `cat ~/.ssh/id_ed25519.pub`) in the "key" box.

# Configure GitHub for your account
To ***configure git*** for your account, enter the following commands in the terminal:
```bash
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```
Optionally, set your pull config to "merge" (the default strategy):
```bash
git config --global pull.rebase false
```


# Extras
The following parts of the tutorial are totally optional and only for those who want to further customize their git installation. If you are not interested in this, you are now done with the installation process. Enjoy using git!

## Install GitHub copilot for the CLI
If you have an activate Copilot subscription, you can use GitHub Copilot for the CLI, which is a tool that can suggest commands based on natural language. Note that this is a paid service, and that our recommendation is solely based on personal preferences. [This](https://docs.github.com/en/copilot/github-copilot-in-the-cli) page describes the installation process, but as usual we have put the necessary commands below for your convenience. Note that the below command requires GitHub CLI to be installed.
```bash
gh extension install github/gh-copilot
```
For shorter aliases, add the following to your `~/.bashrc` file:
```bash
# GitHub Copilot in the CLI: Define aliases for common commands:
alias copilot='gh copilot'
alias '?'='gh copilot explain'
alias '??'='gh copilot suggest -t shell'
alias 'git?'='gh copilot suggest -t git'
alias 'gh?'='gh copilot suggest -t gh'
```
Now you can enter commands like `?? Find all python files containing "import sys"` and Copilot will suggest a Bash command to perform the action.

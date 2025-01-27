# PowerShell
The Windows PowerShell is the standard way of interacting with the Windows environment using a CLI. This guide will present tips and tricks you can use when interacting with PowerShell.


## Exit using Ctrl+D
Sometimes, it may be cumbersome to exit the PowerShell by typing "exit" followed by Enter. Thus, you might want to be able to exit the PowerShell in the same way as the Linux shell (using Ctrl+D). To do this, first run the following command with administrator privileges:
```powershell
Set-ExecutionPolicy RemoteSigned
```
Next, open `$PROFILE` with your favourite editor, for example:
```powershell
code $PROFILE
```
Change the file to contain the following:
```powershell
Set-PSReadlineKeyHandler -Chord Ctrl+d -Function DeleteCharOrExit
```
And save the file. Reopen your PowerShell and now you should be able to press Ctrl+D to exit it!


## Use a Unix-like colored prompt
If you want to able to better recognize when a new command started (for example, this could be handy if your commands have a lot of output), you can change the color of the PowerShell prompt by adding the following to your `$PROFILE`:
```powershell
function prompt {
    $ESC = [char]27

    # Color codes
    $ColorGreen = "$ESC[32m"
    $ColorBlue = "$ESC[34m"
    $ColorWhite = "$ESC[37m"
    $ColorReset = "$ESC[0m"

    # Conda environment
    $condaEnv = $env:CONDA_DEFAULT_ENV
    $condaPrompt = if ($condaEnv) { "$ColorWhite($condaEnv) " } else { "" }

    # Construct the prompt
    "$condaPrompt${ColorGreen}PS $ColorBlue$($executionContext.SessionState.Path.CurrentLocation)$ColorWhite$('>' * ($nestedPromptLevel + 1)) $ColorReset"
}
```


## Tree
To print a tree rooted at 'path/to/directory', you can just use:
```powershell
tree 'path/to/directory'
```

Unfortunately, the default `tree` command does not allow much filtering. Thus, we have made a function that does allow such control for a more fine-grained output:
```powershell
function Print-Tree {
    param (
        [string]$Path,
        [string]$Prefix = '',
        [string]$Match = '',
        [string]$NotMatch = ''
    )

    $items = Get-ChildItem -Path $Path -Recurse -Force

    if ($Match) {
        $items = $items | Where-Object { $_.Name -match $Match }
    }

    if ($NotMatch) {
        $items = $items | Where-Object { $_.Name -notmatch $NotMatch }
    }

    $total = $items.Count

    for ($i = 0; $i -lt $total; $i++) {
        $item = $items[$i]
        $isLast = $i -eq $total - 1
        $marker = if ($isLast) { '└──' } else { '├──' }
        Write-Output "$Prefix$marker $($item.Name)"

        if ($item.PSIsContainer) {
            $newPrefix = if ($isLast) { "$Prefix    " } else { "$Prefix│   " }
            Print-Tree -Path $item.FullName -Prefix $newPrefix -Match $Match -NotMatch $NotMatch
        }
    }
}
```

This function now includes:
1. **`Match` Parameter**: Filters items to include only those whose names match the specified regex. If one of the directories in the path to a file does not match the regex, all files and subdirectories in this directory are also ignored.
2. **`NotMatch` Parameter**: Filters items to exclude those whose names match the specified regex. If one of the directories in the path to a file matches the regex, all files and subdirectories in this directory are also ignored.

Usage examples:
- To print the tree while excluding items starting with `_`:
  ```powershell
  Print-Tree -Path 'path/to/directory' -NotMatch '^_'
  ```
- To print the tree including only items that contain `example` in their names:
  ```powershell
  Print-Tree -Path 'path/to/directory' -Match 'example'
  ```
- To print the tree excluding items starting with `_` and including only items that contain `example` in their names:
  ```powershell
  Print-Tree -Path 'path/to/directory' -Match 'example' -NotMatch '^_'
  ```

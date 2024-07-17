# PowerShell
The Windows PowerShell is the standard way of interacting with the Windows environment using a CLI. This guide will present tips and tricks you can use when interacting with PowerShell.


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

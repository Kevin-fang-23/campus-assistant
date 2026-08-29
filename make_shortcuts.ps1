$ErrorActionPreference = 'Stop'
$ws = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath('Desktop')
$proj = 'C:\Users\86182\WorkBuddy\2026-08-24-17-13-36\campus-assistant'

function New-CampusShortcut {
    param(
        [string]$Name,
        [string]$Bat
    )
    $path = Join-Path $desktop ($Name + '.lnk')
    $lnk = $ws.CreateShortcut($path)
    $lnk.TargetPath = 'C:\Windows\System32\cmd.exe'
    $lnk.Arguments = '/k "cd /d "' + $proj + '" && call ' + $Bat + '"'
    $lnk.WorkingDirectory = $proj
    $lnk.WindowStyle = 1
    $lnk.Description = 'Campus Assistant ' + $Bat
    $lnk.Save()
    Write-Host "created: $path"
}

New-CampusShortcut -Name '校园助手-启动' -Bat 'start.bat'
New-CampusShortcut -Name '校园助手-停止' -Bat 'stop.bat'
Write-Host "Desktop = $desktop"

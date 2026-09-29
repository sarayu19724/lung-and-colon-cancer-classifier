$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Create the virtual environment and install requirements.txt first; see README.md.'
}
$taskWebDependencies = Join-Path $PSScriptRoot '.webdeps'
if (Test-Path -LiteralPath $taskWebDependencies) {
    $env:PYTHONPATH = $taskWebDependencies
}
Write-Host 'TissueLens: http://127.0.0.1:5000 (Ctrl+C to stop)'
& $taskPython app.py

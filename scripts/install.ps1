param(
    [Parameter(Mandatory = $true)]
    [string]$DesktopPath,
    [string]$PackageSource = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [switch]$WithVoiceDependencies
)

$ErrorActionPreference = 'Stop'
$desktop = (Resolve-Path -LiteralPath $DesktopPath).Path
$runtime = Join-Path $desktop 'modules\lerobot-vulcan'
$python = Join-Path $runtime '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Sourccey Desktop Python runtime not found at $python. Run Desktop initial setup first."
}

$requirement = $PackageSource
if ($WithVoiceDependencies) {
    $requirement = "$PackageSource[voice]"
}
uv pip install --python $python -e $requirement
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install sourccey-voice into the Desktop runtime.'
}

$moduleDir = Join-Path $desktop 'modules\sourccey-voice'
New-Item -ItemType Directory -Force -Path $moduleDir | Out-Null
$configTarget = Join-Path $moduleDir 'config.toml'
if (-not (Test-Path -LiteralPath $configTarget)) {
    Copy-Item -LiteralPath (Join-Path $PackageSource 'config\default.toml') -Destination $configTarget
}
Write-Host "Installed Sourccey Voice. Configure $configTarget before starting the host."


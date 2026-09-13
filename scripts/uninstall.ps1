param(
    [Parameter(Mandatory = $true)]
    [string]$DesktopPath,
    [switch]$RemoveConfig
)

$ErrorActionPreference = 'Stop'
$desktop = (Resolve-Path -LiteralPath $DesktopPath).Path
$python = Join-Path $desktop 'modules\lerobot-vulcan\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Sourccey Desktop Python runtime not found at $python."
}
uv pip uninstall --python $python sourccey-voice
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to uninstall sourccey-voice from the Desktop runtime.'
}
if ($RemoveConfig) {
    $moduleDir = (Resolve-Path -LiteralPath (Join-Path $desktop 'modules\sourccey-voice')).Path
    Remove-Item -LiteralPath $moduleDir -Recurse -Force
    Write-Host "Removed package and configuration from $moduleDir."
} else {
    Write-Host 'Removed the package. Configuration and downloaded model caches were preserved.'
}


$ErrorActionPreference = "Stop"

$uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
$ffmpegBin = "C:\Tools\ffmpeg\bin"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot ".uv-python"

if (-not (Test-Path -LiteralPath $uv)) {
    throw "uv was not found at $uv"
}

$env:Path = "$ffmpegBin;$($uv | Split-Path -Parent);$env:Path"
& $uv run --managed-python --python 3.12 python upscaler.py

Set-Location $PSScriptRoot
if (-not (Test-Path .venv\Scripts\pythonw.exe)) {
    uv sync | Out-Null
}
if ($args.Count -eq 0) {
    cmd /c start "" ".venv\Scripts\pythonw.exe" "-m" "upscaler"
} else {
    uv run python -m upscaler @args
}

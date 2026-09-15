Set-Location $PSScriptRoot
if (-not (Test-Path .venv\Scripts\pythonw.exe)) {
    uv sync | Out-Null
}
if ($args.Count -eq 0) {
    Start-Process .venv\Scripts\pythonw.exe -ArgumentList "-m","upscaler" -WindowStyle Hidden
} else {
    uv run python -m upscaler @args
}

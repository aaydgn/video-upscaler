@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
    uv sync >nul 2>&1
)
start "" .venv\Scripts\pythonw.exe -m upscaler %*

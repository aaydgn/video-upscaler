@echo off
cd /d "%~dp0"
uv run python upscaler.py %*
pause

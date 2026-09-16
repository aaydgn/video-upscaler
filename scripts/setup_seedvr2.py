"""Install SeedVR2 standalone upscaler into ~/.video-upscaler/seedvr2.

Creates an isolated venv with PyTorch + SeedVR2 dependencies.
The GGUF model auto-downloads on first inference run (~2 GB).

Usage:
    python scripts/setup_seedvr2.py
    python scripts/setup_seedvr2.py --uninstall
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SEEDVR2_DIR = Path.home() / ".video-upscaler" / "seedvr2"
REPO_URL = "https://github.com/cvtower/SeedVR2_VideoUpscaler_standalone.git"


def setup() -> None:
    if (SEEDVR2_DIR / "inference_cli.py").exists():
        print(f"SeedVR2 already installed at {SEEDVR2_DIR}")
        print("Run with --uninstall first to reinstall.")
        return

    SEEDVR2_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Cloning SeedVR2 standalone into {SEEDVR2_DIR}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(SEEDVR2_DIR)],
        check=True,
    )

    venv_dir = SEEDVR2_DIR / ".venv"
    print(f"Creating venv at {venv_dir}...")
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    pip = str(venv_dir / "Scripts" / "pip.exe") if sys.platform == "win32" else str(venv_dir / "bin" / "pip")
    python = str(venv_dir / "Scripts" / "python.exe") if sys.platform == "win32" else str(venv_dir / "bin" / "python")

    print("Installing PyTorch with CUDA...")
    subprocess.run(
        [pip, "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"],
        check=True,
    )

    print("Installing SeedVR2 dependencies...")
    subprocess.run(
        [pip, "install", "-r", str(SEEDVR2_DIR / "requirements.txt")],
        check=True,
    )

    print(f"\nSeedVR2 installed successfully.")
    print(f"  Location: {SEEDVR2_DIR}")
    print(f"  Python:   {python}")
    print(f"  The GGUF model (~2 GB) will download automatically on first run.")


def uninstall() -> None:
    if SEEDVR2_DIR.exists():
        print(f"Removing {SEEDVR2_DIR}...")
        shutil.rmtree(SEEDVR2_DIR, ignore_errors=True)
        print("Uninstalled.")
    else:
        print("SeedVR2 is not installed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install SeedVR2 standalone upscaler")
    parser.add_argument("--uninstall", action="store_true", help="Remove SeedVR2 installation")
    args = parser.parse_args()
    if args.uninstall:
        uninstall()
    else:
        setup()

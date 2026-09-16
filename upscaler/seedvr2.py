from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Callable

from upscaler.config import ProgressCallback
from upscaler.process import active_subprocess, report_progress


SEEDVR2_DIR = Path.home() / ".video-upscaler" / "seedvr2"


def find_seedvr2() -> tuple[str, str] | None:
    """Find SeedVR2 installation. Returns (python_path, cli_path) or None."""
    cli = SEEDVR2_DIR / "inference_cli.py"
    if not cli.exists():
        return None
    if os.name == "nt":
        python = SEEDVR2_DIR / ".venv" / "Scripts" / "python.exe"
    else:
        python = SEEDVR2_DIR / ".venv" / "bin" / "python"
    if not python.exists():
        return None
    return str(python), str(cli)


def run_seedvr2(
    input_path: Path,
    output_path: Path,
    resolution: int,
    overwrite: bool,
    log: Callable[[str], None],
    progress: ProgressCallback | None,
) -> int:
    found = find_seedvr2()
    if found is None:
        raise RuntimeError(
            "SeedVR2 is not installed. Run:\n"
            "  python scripts/setup_seedvr2.py\n"
            "to install it."
        )

    python, cli = found

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    cmd = [
        python, cli,
        str(input_path),
        "--output", str(output_path),
        "--resolution", str(resolution),
        "--batch_size", "5",
        "--blocks_to_swap", "28",
        "--vae_encode_tiled",
        "--vae_decode_tiled",
        "--color_correction", "wavelet",
        "--seed", "42",
    ]

    log(subprocess.list2cmdline(cmd))
    report_progress("SeedVR2 upscaling", None, None, log, progress)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(SEEDVR2_DIR),
    )
    if process.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    active_subprocess[0] = process

    total_frames: int | None = None
    frame_re = re.compile(r"(\d+)\s*/\s*(\d+)\s*frames?", re.IGNORECASE)
    fps_re = re.compile(r"Average FPS:\s*([\d.]+)", re.IGNORECASE)

    try:
        for line in process.stdout:
            text = line.rstrip()
            if not text:
                continue
            log(f"[SeedVR2] {text}")

            m = frame_re.search(text)
            if m:
                current = int(m.group(1))
                total_frames = int(m.group(2))
                report_progress(
                    "SeedVR2 upscaling", current, total_frames, log, progress,
                )

            m = fps_re.search(text)
            if m:
                log(f"SeedVR2 average speed: {m.group(1)} fps")

        return_code = process.wait()
    finally:
        active_subprocess[0] = None

    return return_code

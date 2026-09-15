from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from upscaler.config import ProgressCallback

# Holds the currently running subprocess so the GUI can terminate it on cancel.
active_subprocess: list[subprocess.Popen | None] = [None]


def stream_command(args: list[str], log: Callable[[str], None], cwd: str | None = None) -> int:
    log(subprocess.list2cmdline(args))
    process = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    active_subprocess[0] = process
    try:
        for line in process.stdout:
            log(line.rstrip())
        return process.wait()
    finally:
        active_subprocess[0] = None


def count_image_files(directory: Path) -> int:
    try:
        return sum(
            1
            for entry in os.scandir(directory)
            if entry.is_file() and entry.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        )
    except OSError:
        return 0


def report_progress(
    phase: str,
    current: int | None,
    total: int | None,
    log: Callable[[str], None],
    progress: ProgressCallback | None,
) -> None:
    if progress:
        progress(phase, current, total)
    if current is not None and total:
        percent = min(100.0, (current / total) * 100)
        log(f"{phase}: {current} / {total} frames ({percent:.1f}%)")
    else:
        log(f"{phase}...")


def stream_command_with_frame_progress(
    args: list[str],
    log: Callable[[str], None],
    output_dir: Path,
    total_frames: int,
    progress: ProgressCallback | None,
    phase: str = "AI upscaling frames",
    log_prefix: str = "Real-ESRGAN",
    cwd: str | None = None,
) -> int:
    log(subprocess.list2cmdline(args))
    stop_event = threading.Event()
    lock = threading.Lock()
    last_reported = {"count": -1, "time": 0.0}

    def emit(force: bool = False) -> None:
        current = count_image_files(output_dir)
        now = time.monotonic()
        with lock:
            changed = current != last_reported["count"]
            due = now - last_reported["time"] >= 2.0
            if force and not changed and last_reported["count"] != -1:
                return
            if not force and not (changed and (due or current == total_frames)):
                return
            last_reported["count"] = current
            last_reported["time"] = now
        report_progress(phase, current, total_frames, log, progress)

    def monitor() -> None:
        emit(force=True)
        while not stop_event.wait(1.0):
            emit()
        emit(force=True)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    process = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    active_subprocess[0] = process
    try:
        for line in process.stdout:
            text = line.rstrip()
            if text:
                log(f"[{log_prefix}] {text}")
        return_code = process.wait()
    finally:
        active_subprocess[0] = None
    stop_event.set()
    monitor_thread.join(timeout=5)
    return return_code

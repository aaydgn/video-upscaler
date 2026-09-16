from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

from upscaler.config import AI_2X_MODEL
from upscaler.engines import upscale
from upscaler.models import DEFAULT_ONNX_MODEL, FRIENDLY_NAMES, list_installed
from upscaler.process import active_subprocess
from upscaler.tools import inspect_tools


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("NVIDIA Video Upscaler")
    root.geometry("620x520")
    root.minsize(520, 460)

    input_var = tk.StringVar()
    output_var = tk.StringVar()
    ai_scale_var = tk.IntVar(value=1)
    ai_scale_label_var = tk.StringVar(value="Off")
    model_var = tk.StringVar(value="Animation")
    backend_var = tk.StringVar(value="Auto")
    codec_var = tk.StringVar(value="h264")
    quality_var = tk.IntVar(value=19)
    enhance_var = tk.BooleanVar(value=True)
    interp60_var = tk.BooleanVar(value=False)
    progress_var = tk.DoubleVar(value=0.0)
    progress_text_var = tk.StringVar(value="Idle")
    elapsed_var = tk.StringVar(value="")
    tool_status_var = tk.StringVar(value="Checking tools…")
    messages: queue.Queue = queue.Queue()

    scale_labels = {1: "Off", 2: "2x", 3: "3x", 4: "4x"}
    ncnn_models = {
        "General": "realesrgan-x4plus",
        "Animation": "realesr-animevideov3",
        "Anime": "realesrgan-x4plus-anime",
        "Fast": "realesrnet-x4plus",
    }
    backends = {"Auto": "auto", "ONNX (pipe)": "onnx", "ncnn (legacy)": "ncnn"}
    onnx_model_var = tk.StringVar()
    _onnx_models: dict[str, str] = {}

    def _refresh_onnx_models(scale: int) -> None:
        _onnx_models.clear()
        for name, info in list_installed():
            if info.scale == scale:
                friendly = FRIENDLY_NAMES.get(name, name)
                _onnx_models[friendly] = name
        if not _onnx_models:
            for name, info in list_installed():
                if info.scale >= scale:
                    friendly = FRIENDLY_NAMES.get(name, name)
                    _onnx_models[friendly] = name

    _job_start_time: list[float | None] = [None]
    _elapsed_after_id: list[str | None] = [None]
    _last_output_path: list[str] = [""]
    _cancel_event = threading.Event()

    _prefs_path = Path.home() / ".video-upscaler-prefs.json"

    def load_settings() -> None:
        try:
            prefs = json.loads(_prefs_path.read_text(encoding="utf-8"))
            if prefs.get("codec") in ("h264", "hevc"):
                codec_var.set(prefs["codec"])
            if isinstance(prefs.get("quality"), int):
                quality_var.set(max(14, min(28, prefs["quality"])))
            if isinstance(prefs.get("ai_scale"), int) and prefs["ai_scale"] in scale_labels:
                ai_scale_var.set(prefs["ai_scale"])
            if prefs.get("model") in ncnn_models:
                model_var.set(prefs["model"])
            if prefs.get("backend") in backends:
                backend_var.set(prefs["backend"])
            if isinstance(prefs.get("enhance"), bool):
                enhance_var.set(prefs["enhance"])
            if isinstance(prefs.get("interp60"), bool):
                interp60_var.set(prefs["interp60"])
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    def save_settings() -> None:
        try:
            prefs = {
                "codec": codec_var.get(),
                "quality": quality_var.get(),
                "ai_scale": ai_scale_var.get(),
                "model": model_var.get(),
                "backend": backend_var.get(),
                "enhance": enhance_var.get(),
                "interp60": interp60_var.get(),
            }
            _prefs_path.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
        except OSError:
            pass

    def on_close() -> None:
        save_settings()
        root.destroy()

    class Tooltip:
        def __init__(self, widget: tk.Widget, text: str) -> None:
            self.widget = widget
            self.text = text
            self.tip: tk.Toplevel | None = None
            widget.bind("<Enter>", self.show)
            widget.bind("<Leave>", self.hide)

        def show(self, _event: tk.Event) -> None:
            if self.tip:
                return
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
            self.tip = tk.Toplevel(self.widget)
            self.tip.wm_overrideredirect(True)
            self.tip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                self.tip,
                text=self.text,
                justify="left",
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                padx=8,
                pady=5,
                wraplength=340,
            )
            label.pack()

        def hide(self, _event: tk.Event) -> None:
            if self.tip:
                self.tip.destroy()
                self.tip = None

    def default_output_for(path: Path) -> Path:
        scale = ai_scale_var.get()
        parts = []
        if scale > 1:
            parts.append(f"ai{scale}x")
        if enhance_var.get():
            parts.append("enhanced")
        if interp60_var.get():
            parts.append("60fps")
        suffix = "_".join(parts) if parts else "encoded"
        return path.with_name(f"{path.stem}_{suffix}.mp4")

    def update_default_output(*_args: object) -> None:
        if input_var.get():
            output_var.set(str(default_output_for(Path(input_var.get()))))

    def _is_onnx_backend() -> bool:
        key = backend_var.get()
        return backends.get(key, "auto") in ("onnx", "auto")

    def _sync_model_combo() -> None:
        val = ai_scale_var.get()
        if val <= 1:
            model_label.grid_remove()
            model_combo.grid_remove()
            model_combo.configure(state="disabled")
            return
        if _is_onnx_backend():
            _refresh_onnx_models(val)
            model_combo.configure(values=tuple(_onnx_models.keys()))
            if model_var.get() not in _onnx_models:
                first = next(iter(_onnx_models), "")
                model_var.set(first)
            model_label.grid()
            model_combo.grid()
            model_combo.configure(state="readonly")
        elif val >= 4:
            model_combo.configure(values=tuple(ncnn_models.keys()))
            if model_var.get() not in ncnn_models:
                model_var.set("Animation")
            model_label.grid()
            model_combo.grid()
            model_combo.configure(state="readonly")
        else:
            model_label.grid_remove()
            model_combo.grid_remove()
            model_combo.configure(state="disabled")

    def on_scale_change(_value: str) -> None:
        val = round(float(_value))
        ai_scale_var.set(val)
        ai_scale_label_var.set(scale_labels[val])
        if val > 1:
            backend_label.grid()
            backend_combo.grid()
            backend_combo.configure(state="readonly")
        else:
            backend_label.grid_remove()
            backend_combo.grid_remove()
            backend_combo.configure(state="disabled")
        _sync_model_combo()
        update_default_output()

    def on_backend_change(_event: object = None) -> None:
        _sync_model_combo()

    def on_quality_change(_value: str) -> None:
        quality_var.set(round(float(_value)))

    def choose_input() -> None:
        filename = filedialog.askopenfilename(
            title="Choose source video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            input_var.set(filename)
            update_default_output()

    def choose_output() -> None:
        filename = filedialog.asksaveasfilename(
            title="Save output video as",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
        )
        if filename:
            output_var.set(filename)

    def append_log(text: str) -> None:
        log_box.configure(state="normal")
        log_box.insert("end", text + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def queue_progress(phase: str, current: int | None, total: int | None) -> None:
        messages.put(("progress", phase, current, total))

    def apply_progress(phase: str, current: int | None, total: int | None) -> None:
        if current is not None and total:
            percent = min(100.0, (current / total) * 100)
            progress_var.set(percent)
            progress_text_var.set(f"{phase}: {current} / {total} frames ({percent:.1f}%)")
        elif phase == "Done":
            progress_var.set(100.0)
            progress_text_var.set("Done")
        else:
            progress_text_var.set(phase)

    def set_controls_enabled(enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        input_entry.configure(state=state)
        output_entry.configure(state=state)
        input_browse_btn.configure(state=state)
        output_browse_btn.configure(state=state)
        ai_scale_slider.configure(state=state)
        model_combo.configure(state="readonly" if enabled and ai_scale_var.get() >= 4 else "disabled")
        backend_combo.configure(state="readonly" if enabled and ai_scale_var.get() > 1 else "disabled")
        for rb in codec_radios:
            rb.configure(state=state)
        quality_scale.configure(state=state)
        if enabled:
            enhance_check.state(["!disabled"])
            interp60_check.state(["!disabled"])
        else:
            enhance_check.state(["disabled"])
            interp60_check.state(["disabled"])

    def tick_elapsed() -> None:
        if _job_start_time[0] is None:
            return
        elapsed = int(time.monotonic() - _job_start_time[0])
        m, s = divmod(elapsed, 60)
        elapsed_var.set(f"  |  Elapsed: {m}m {s:02d}s")
        _elapsed_after_id[0] = root.after(1000, tick_elapsed)

    def stop_elapsed() -> None:
        if _elapsed_after_id[0]:
            root.after_cancel(_elapsed_after_id[0])
            _elapsed_after_id[0] = None
        _job_start_time[0] = None

    def on_job_done(success: bool, output_path_str: str) -> None:
        stop_elapsed()
        set_controls_enabled(True)
        start_button.grid()
        cancel_button.grid_remove()
        if success and output_path_str:
            _last_output_path[0] = output_path_str
            reveal_button.grid()
        else:
            reveal_button.grid_remove()

    def open_output_folder() -> None:
        p = Path(_last_output_path[0])
        folder = p.parent if p.is_file() else p
        try:
            os.startfile(str(folder))
        except (OSError, AttributeError):
            messagebox.showinfo("Output folder", str(folder))

    def cancel_job() -> None:
        _cancel_event.set()
        proc = active_subprocess[0]
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass

    def drain_messages() -> None:
        try:
            while True:
                item = messages.get_nowait()
                if isinstance(item, tuple) and item:
                    kind = item[0]
                    if kind == "progress":
                        try:
                            _, phase, current, total = item
                            apply_progress(phase, current, total)
                        except (ValueError, TypeError):
                            append_log(str(item))
                    elif kind == "done":
                        try:
                            _, success, output_path_str = item
                            on_job_done(success, output_path_str)
                        except (ValueError, TypeError):
                            pass
                    elif kind == "tool_status":
                        try:
                            _, status_text = item
                            tool_status_var.set(status_text)
                        except (ValueError, TypeError):
                            pass
                    else:
                        append_log(str(item))
                else:
                    append_log(str(item))
        except queue.Empty:
            pass
        root.after(100, drain_messages)

    def start() -> None:
        if not input_var.get():
            messagebox.showerror("Missing input", "Choose a source video first.")
            return
        input_path = Path(input_var.get())
        if not input_path.exists():
            messagebox.showerror("Not found", f"Input file not found:\n{input_path}")
            return
        out_str = output_var.get() or str(default_output_for(input_path))
        output_path = Path(out_str)

        if output_path.exists():
            if not messagebox.askyesno("File exists", f"Overwrite?\n{output_path.name}"):
                return

        scale = ai_scale_var.get()
        engine = "ai" if scale > 1 else "ffmpeg"
        model_key = model_var.get()
        model = ncnn_models.get(model_key, AI_2X_MODEL)
        backend_key = backend_var.get()
        backend = backends.get(backend_key, "auto")
        onnx_model_name = _onnx_models.get(model_key, DEFAULT_ONNX_MODEL)

        _cancel_event.clear()
        set_controls_enabled(False)
        start_button.grid_remove()
        cancel_button.grid()
        reveal_button.grid_remove()
        progress_var.set(0.0)
        progress_text_var.set("Starting")
        elapsed_var.set("")
        log_box.configure(state="normal")
        log_box.delete("1.0", "end")
        log_box.configure(state="disabled")
        _job_start_time[0] = time.monotonic()
        tick_elapsed()

        def worker() -> None:
            success = False
            try:
                code = upscale(
                    str(input_path),
                    out_str,
                    engine,
                    model,
                    scale,
                    codec_var.get(),
                    quality_var.get(),
                    True,
                    enhance_var.get(),
                    interp60_var.get(),
                    None,
                    backend=backend,
                    onnx_model=onnx_model_name,
                    log=messages.put,
                    progress=queue_progress,
                )
                success = code == 0
                if not success and not _cancel_event.is_set():
                    messages.put(f"Upscale failed. Output was: {out_str}\nCheck the log above.")
                elif _cancel_event.is_set():
                    messages.put("Cancelled.")
            except Exception as exc:
                messages.put(f"Error: {exc}")
            finally:
                messages.put(("done", success, out_str if success else ""))

        threading.Thread(target=worker, daemon=True).start()

    def check_tools_background() -> None:
        try:
            tools = inspect_tools()
            parts = ["FFmpeg ✓"]
            nvenc = "H.264+HEVC" if tools.has_h264_nvenc and tools.has_hevc_nvenc else "H.264" if tools.has_h264_nvenc else "HEVC" if tools.has_hevc_nvenc else None
            parts.append(f"NVENC {nvenc}" if nvenc else "NVENC ✗")
            parts.append("ESRGAN ✓" if tools.realesrgan else "ESRGAN ✗")
            parts.append("RIFE ✓" if tools.rife else "RIFE ✗")
            try:
                import onnxruntime as ort
                from upscaler.onnx_upscale import select_provider
                parts.append(f"ONNX ✓ ({select_provider().replace('ExecutionProvider', '')})")
            except ImportError:
                parts.append("ONNX ✗")
            messages.put(("tool_status", "  ·  ".join(parts)))
        except Exception as exc:
            messages.put(("tool_status", f"Tool check failed: {exc}"))

    threading.Thread(target=check_tools_background, daemon=True).start()

    # ------------------------------------------------------------------ layout
    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    row = 0

    # --- Input ---
    ttk.Label(frame, text="Input").grid(row=row, column=0, sticky="w", pady=4)
    input_entry = ttk.Entry(frame, textvariable=input_var)
    input_entry.grid(row=row, column=1, sticky="ew", padx=8)
    input_browse_btn = ttk.Button(frame, text="Browse", command=choose_input)
    input_browse_btn.grid(row=row, column=2)

    row += 1
    ttk.Label(frame, text="Save as").grid(row=row, column=0, sticky="w", pady=4)
    output_entry = ttk.Entry(frame, textvariable=output_var)
    output_entry.grid(row=row, column=1, sticky="ew", padx=8)
    output_browse_btn = ttk.Button(frame, text="Browse", command=choose_output)
    output_browse_btn.grid(row=row, column=2)

    row += 1
    ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)

    # --- AI upscale slider ---
    row += 1
    ttk.Label(frame, text="AI upscale").grid(row=row, column=0, sticky="w", pady=4)
    scale_frame = ttk.Frame(frame)
    scale_frame.grid(row=row, column=1, sticky="ew", padx=8)
    scale_frame.columnconfigure(0, weight=1)
    ai_scale_slider = ttk.Scale(scale_frame, from_=1, to=4, variable=ai_scale_var, orient="horizontal", command=on_scale_change)
    ai_scale_slider.grid(row=0, column=0, sticky="ew")
    ttk.Label(scale_frame, textvariable=ai_scale_label_var, width=4, anchor="e").grid(row=0, column=1, padx=(8, 0))
    Tooltip(ai_scale_slider, "Off: no AI upscaling.\n2x/3x/4x: Real-ESRGAN upscale factor.\nOutput resolution = input × scale.")

    # --- AI model (visible only when scale > 1) ---
    row += 1
    model_label = ttk.Label(frame, text="AI model")
    model_label.grid(row=row, column=0, sticky="w", pady=4)
    model_combo = ttk.Combobox(
        frame, textvariable=model_var,
        values=tuple(ncnn_models.keys()), state="disabled", width=16,
    )
    model_combo.grid(row=row, column=1, sticky="w", padx=8)
    model_label.grid_remove()
    model_combo.grid_remove()
    Tooltip(model_combo, "General: best for live action and photos.\nAnimation: optimized for animated video.\nAnime: tuned for anime art style.\nFast: lighter model, quicker but lower quality.")

    # --- Backend ---
    row += 1
    backend_label = ttk.Label(frame, text="Backend")
    backend_label.grid(row=row, column=0, sticky="w", pady=4)
    backend_combo = ttk.Combobox(
        frame, textvariable=backend_var,
        values=tuple(backends.keys()), state="disabled", width=16,
    )
    backend_combo.grid(row=row, column=1, sticky="w", padx=8)
    backend_combo.bind("<<ComboboxSelected>>", on_backend_change)
    backend_label.grid_remove()
    backend_combo.grid_remove()
    Tooltip(backend_combo, "Auto: use ONNX if installed, else ncnn.\nONNX (pipe): in-process inference, zero disk I/O.\nncnn (legacy): realesrgan-ncnn-vulkan binary.")

    # --- Options ---
    row += 1
    options_frame = ttk.Frame(frame)
    options_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 0))
    enhance_check = ttk.Checkbutton(options_frame, text="Enhance", variable=enhance_var, command=update_default_output)
    enhance_check.pack(side="left", padx=(0, 16))
    Tooltip(enhance_check, "Deblock, denoise, and CAS sharpen.\nFor AI upscale, applied as a pre-filter.")
    interp60_check = ttk.Checkbutton(options_frame, text="60 fps", variable=interp60_var, command=update_default_output)
    interp60_check.pack(side="left")
    Tooltip(interp60_check, "RIFE GPU interpolation to exact 60 fps.")

    row += 1
    ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)

    # --- Encoding ---
    row += 1
    ttk.Label(frame, text="Codec").grid(row=row, column=0, sticky="w", pady=4)
    codec_frame = ttk.Frame(frame)
    codec_frame.grid(row=row, column=1, sticky="w", padx=8)
    codec_radios = [
        ttk.Radiobutton(codec_frame, text="H.264", value="h264", variable=codec_var),
        ttk.Radiobutton(codec_frame, text="HEVC", value="hevc", variable=codec_var),
    ]
    codec_radios[0].pack(side="left")
    codec_radios[1].pack(side="left", padx=16)

    row += 1
    ttk.Label(frame, text="Quality").grid(row=row, column=0, sticky="w", pady=4)
    quality_frame = ttk.Frame(frame)
    quality_frame.grid(row=row, column=1, sticky="ew", padx=8)
    quality_frame.columnconfigure(0, weight=1)
    quality_scale = ttk.Scale(quality_frame, from_=14, to=28, variable=quality_var, orient="horizontal", command=on_quality_change)
    quality_scale.grid(row=0, column=0, sticky="ew")
    ttk.Label(quality_frame, textvariable=quality_var, width=3, anchor="e").grid(row=0, column=1, padx=(8, 0))
    Tooltip(quality_scale, "NVENC constant quality. Lower = better quality, larger file.\n14 = near-lossless, 19 = balanced, 28 = small file.")

    # --- Action + progress ---
    row += 1
    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 0))
    start_button = ttk.Button(btn_frame, text="Process video", command=start)
    start_button.grid(row=0, column=0)
    cancel_button = ttk.Button(btn_frame, text="Cancel", command=cancel_job)
    cancel_button.grid(row=0, column=0)
    cancel_button.grid_remove()
    reveal_button = ttk.Button(btn_frame, text="Open folder", command=open_output_folder)
    reveal_button.grid(row=0, column=1, padx=(12, 0))
    reveal_button.grid_remove()

    row += 1
    progress_bar = ttk.Progressbar(frame, variable=progress_var, maximum=100, mode="determinate")
    progress_bar.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 2))

    row += 1
    status_frame = ttk.Frame(frame)
    status_frame.grid(row=row, column=0, columnspan=3, sticky="ew")
    ttk.Label(status_frame, textvariable=progress_text_var).pack(side="left")
    ttk.Label(status_frame, textvariable=elapsed_var, foreground="gray").pack(side="left")

    row += 1
    frame.rowconfigure(row, weight=1)
    log_box = tk.Text(frame, height=8, state="disabled", wrap="word")
    log_box.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(8, 0))

    # --- Status bar ---
    tool_bar = ttk.Label(root, textvariable=tool_status_var, foreground="gray", padding=(16, 4))
    tool_bar.pack(side="bottom", fill="x")

    root.protocol("WM_DELETE_WINDOW", on_close)
    load_settings()
    on_scale_change(str(ai_scale_var.get()))
    drain_messages()
    root.mainloop()

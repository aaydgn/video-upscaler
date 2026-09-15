from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

from upscaler.config import AI_2X_MODEL
from upscaler.engines import upscale
from upscaler.process import active_subprocess
from upscaler.tools import inspect_tools


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("NVIDIA Video Upscaler")
    root.geometry("760x580")
    root.minsize(680, 520)

    input_var = tk.StringVar()
    output_var = tk.StringVar()
    workflow_var = tk.StringVar(value="Enhance")
    model_var = tk.StringVar(value=AI_2X_MODEL)
    codec_var = tk.StringVar(value="h264")
    quality_var = tk.IntVar(value=19)
    enhance_var = tk.BooleanVar(value=True)
    interp60_var = tk.BooleanVar(value=False)
    overwrite_var = tk.BooleanVar(value=False)
    progress_var = tk.DoubleVar(value=0.0)
    progress_text_var = tk.StringVar(value="Idle")
    elapsed_var = tk.StringVar(value="")
    tool_status_var = tk.StringVar(value="Checking tools…")
    messages: queue.Queue = queue.Queue()

    workflows = {
        "Enhance": "ffmpeg",
        "AI upscale 2x": "ai",
        "Interpolate to 60 fps": "interp60",
    }
    ai_workflows = {"AI upscale 2x"}

    _job_start_time: list[float | None] = [None]
    _elapsed_after_id: list[str | None] = [None]
    _last_output_path: list[str] = [""]
    _cancel_event = threading.Event()

    _prefs_path = Path.home() / ".video-upscaler-prefs.json"

    def load_settings() -> None:
        try:
            prefs = json.loads(_prefs_path.read_text(encoding="utf-8"))
            if prefs.get("workflow") in workflows:
                workflow_var.set(prefs["workflow"])
            if prefs.get("codec") in ("h264", "hevc"):
                codec_var.set(prefs["codec"])
            if isinstance(prefs.get("quality"), int):
                quality_var.set(max(14, min(28, prefs["quality"])))
            if isinstance(prefs.get("enhance"), bool):
                enhance_var.set(prefs["enhance"])
            if isinstance(prefs.get("interp60"), bool):
                interp60_var.set(prefs["interp60"])
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    def save_settings() -> None:
        try:
            prefs = {
                "workflow": workflow_var.get(),
                "codec": codec_var.get(),
                "quality": quality_var.get(),
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
        workflow = workflow_var.get()
        interpolate = interp60_var.get() and workflow != "Interpolate to 60 fps"
        if workflow == "Interpolate to 60 fps":
            suffix = "60fps"
        elif workflow == "AI upscale 2x":
            suffix = "ai2x"
        elif enhance_var.get():
            suffix = "enhanced"
        else:
            suffix = "encoded"
        if interpolate:
            suffix += "_60fps"
        return path.with_name(f"{path.stem}_{suffix}.mp4")

    def update_default_output(_event: tk.Event | None = None) -> None:
        if input_var.get():
            output_var.set(str(default_output_for(Path(input_var.get()))))

    def update_workflow_controls(_event: tk.Event | None = None) -> None:
        workflow = workflow_var.get()
        if workflow == "Interpolate to 60 fps":
            interp60_var.set(False)
            interp60_check.state(["disabled"])
            enhance_check.state(["disabled"])
        else:
            interp60_check.state(["!disabled"])
            enhance_check.state(["!disabled"])
        model_combo.configure(state="readonly" if workflow in ai_workflows else "disabled")
        start_button.configure(
            text="Interpolate to 60 fps" if workflow == "Interpolate to 60 fps" else "Process video"
        )
        update_default_output()

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
            output_var.set(str(default_output_for(Path(filename))))

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
        workflow_combo.configure(state="readonly" if enabled else "disabled")
        model_combo.configure(
            state="readonly" if enabled and workflow_var.get() in ai_workflows else "disabled"
        )
        for rb in codec_radios:
            rb.configure(state=state)
        quality_scale.configure(state=state)
        overwrite_check.configure(state=state)
        if enabled:
            workflow = workflow_var.get()
            if workflow == "Interpolate to 60 fps":
                interp60_check.state(["disabled"])
                enhance_check.state(["disabled"])
            else:
                interp60_check.state(["!disabled"])
                enhance_check.state(["!disabled"])
        else:
            interp60_check.state(["disabled"])
            enhance_check.state(["disabled"])

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

    def validate_inputs(input_path: Path, output_path: Path) -> list[str]:
        warnings: list[str] = []
        if not input_path.exists():
            warnings.append(f"Input file not found:\n{input_path}")
            return warnings
        if not input_path.is_file():
            warnings.append(f"Input is not a file:\n{input_path}")
            return warnings
        out_dir = output_path.parent
        if not out_dir.exists():
            warnings.append(f"Output directory does not exist:\n{out_dir}")
        return warnings

    def start() -> None:
        if not input_var.get():
            messagebox.showerror("Missing input", "Choose a source video first.")
            return
        selected_workflow = workflow_var.get()
        selected_engine = workflows[selected_workflow]
        input_path = Path(input_var.get())
        out_str = output_var.get() or str(default_output_for(input_path))
        output_path = Path(out_str)

        warnings = validate_inputs(input_path, output_path)
        if warnings:
            msg = "\n\n".join(warnings) + "\n\nProceed anyway?"
            if not messagebox.askokcancel("Validation warnings", msg):
                return

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
                    selected_engine,
                    model_var.get(),
                    codec_var.get(),
                    quality_var.get(),
                    overwrite_var.get(),
                    enhance_var.get(),
                    interp60_var.get() and selected_workflow != "Interpolate to 60 fps",
                    None,
                    messages.put,
                    queue_progress,
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
            parts = []
            parts.append("FFmpeg: OK")
            parts.append(f"NVENC: {'H.264+HEVC' if tools.has_h264_nvenc and tools.has_hevc_nvenc else 'H.264' if tools.has_h264_nvenc else 'HEVC' if tools.has_hevc_nvenc else 'NOT FOUND'}")
            parts.append(f"Real-ESRGAN: {'OK' if tools.realesrgan else 'not found (AI workflows unavailable)'}")
            parts.append(f"RIFE: {'OK' if tools.rife else 'not found (interpolation unavailable)'}")
            messages.put(("tool_status", "  ".join(parts)))
        except Exception as exc:
            messages.put(("tool_status", f"Tool check failed: {exc}"))

    threading.Thread(target=check_tools_background, daemon=True).start()

    # ------------------------------------------------------------------ layout
    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(13, weight=1)

    ttk.Label(frame, text="Input video").grid(row=0, column=0, sticky="w", pady=4)
    input_entry = ttk.Entry(frame, textvariable=input_var)
    input_entry.grid(row=0, column=1, sticky="ew", padx=8)
    input_browse_btn = ttk.Button(frame, text="Browse", command=choose_input)
    input_browse_btn.grid(row=0, column=2)

    ttk.Label(frame, text="Output video").grid(row=1, column=0, sticky="w", pady=4)
    output_entry = ttk.Entry(frame, textvariable=output_var)
    output_entry.grid(row=1, column=1, sticky="ew", padx=8)
    output_browse_btn = ttk.Button(frame, text="Browse", command=choose_output)
    output_browse_btn.grid(row=1, column=2)

    ttk.Label(frame, text="Codec").grid(row=2, column=0, sticky="w", pady=4)
    codec_frame = ttk.Frame(frame)
    codec_frame.grid(row=2, column=1, sticky="w", padx=8)
    codec_radios = [
        ttk.Radiobutton(codec_frame, text="H.264 NVENC", value="h264", variable=codec_var),
        ttk.Radiobutton(codec_frame, text="HEVC NVENC", value="hevc", variable=codec_var),
    ]
    codec_radios[0].pack(side="left")
    codec_radios[1].pack(side="left", padx=16)

    ttk.Label(frame, text="Workflow").grid(row=3, column=0, sticky="w", pady=4)
    workflow_combo = ttk.Combobox(frame, textvariable=workflow_var, values=tuple(workflows.keys()), state="readonly")
    workflow_combo.grid(row=3, column=1, sticky="ew", padx=8)
    workflow_combo.bind("<<ComboboxSelected>>", update_workflow_controls)
    Tooltip(
        workflow_combo,
        "Enhance: FFmpeg deblock/denoise/sharpen filters.\n"
        "AI upscale 2x: Real-ESRGAN 2x upscale from any input resolution.\n"
        "Interpolate to 60 fps: RIFE frame interpolation, preserves resolution.",
    )

    ttk.Label(frame, textvariable=tool_status_var, foreground="gray").grid(
        row=4, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4)
    )

    ttk.Label(frame, text="AI model").grid(row=5, column=0, sticky="w", pady=4)
    model_combo = ttk.Combobox(
        frame,
        textvariable=model_var,
        values=("realesrgan-x4plus", "realesr-animevideov3", "realesrgan-x4plus-anime", "realesrnet-x4plus"),
        state="disabled",
    )
    model_combo.grid(row=5, column=1, sticky="ew", padx=8)

    ttk.Label(frame, text="Quality").grid(row=6, column=0, sticky="w", pady=4)
    quality_scale = ttk.Scale(frame, from_=14, to=28, variable=quality_var, orient="horizontal")
    quality_scale.grid(row=6, column=1, sticky="ew", padx=8)
    ttk.Label(frame, textvariable=quality_var, width=4).grid(row=6, column=2, sticky="w")

    enhance_check = ttk.Checkbutton(
        frame, text="Enhance (deblock/denoise/sharpen)", variable=enhance_var, command=update_default_output
    )
    enhance_check.grid(row=7, column=1, sticky="w", padx=8, pady=4)
    Tooltip(
        enhance_check,
        "Apply deblocking, denoising, and CAS sharpening.\n"
        "For AI workflows, applies deblock/denoise as a pre-filter before upscaling.",
    )

    interp60_check = ttk.Checkbutton(
        frame, text="Interpolate to 60 fps", variable=interp60_var, command=update_default_output
    )
    interp60_check.grid(row=8, column=1, sticky="w", padx=8, pady=4)
    Tooltip(
        interp60_check,
        "Uses RIFE on the GPU to produce exact 60 fps. "
        "The app extracts frames, runs rife-ncnn-vulkan, then reassembles with NVENC.",
    )

    overwrite_check = ttk.Checkbutton(frame, text="Overwrite output if it exists", variable=overwrite_var)
    overwrite_check.grid(row=9, column=1, sticky="w", padx=8, pady=4)

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=10, column=0, columnspan=3, sticky="w", padx=8, pady=10)
    start_button = ttk.Button(btn_frame, text="Process video", command=start)
    start_button.grid(row=0, column=0)
    cancel_button = ttk.Button(btn_frame, text="Cancel", command=cancel_job)
    cancel_button.grid(row=0, column=0)
    cancel_button.grid_remove()
    reveal_button = ttk.Button(btn_frame, text="Open folder", command=open_output_folder)
    reveal_button.grid(row=0, column=1, padx=(12, 0))
    reveal_button.grid_remove()

    progress_bar = ttk.Progressbar(frame, variable=progress_var, maximum=100, mode="determinate")
    progress_bar.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(4, 2))

    status_frame = ttk.Frame(frame)
    status_frame.grid(row=12, column=0, columnspan=3, sticky="ew")
    ttk.Label(status_frame, textvariable=progress_text_var).pack(side="left")
    ttk.Label(status_frame, textvariable=elapsed_var, foreground="gray").pack(side="left")

    log_box = tk.Text(frame, height=10, state="disabled", wrap="word")
    log_box.grid(row=13, column=0, columnspan=3, sticky="nsew", pady=(8, 0))

    root.protocol("WM_DELETE_WINDOW", on_close)
    load_settings()
    update_workflow_controls()
    drain_messages()
    root.mainloop()

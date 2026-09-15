# NVIDIA Video Upscaler

Small Python/Tkinter wrapper for video enhancement and AI upscaling with
FFmpeg/NVENC and optional Real-ESRGAN and RIFE. The default workflow applies
deblocking, denoising, and CAS sharpening, then encodes with NVENC.

## Requirements

- NVIDIA RTX GPU and current NVIDIA driver
- `uv`
- FFmpeg with NVENC support on `PATH`
- Real-ESRGAN ncnn Vulkan, optional but required for AI upscaling
- RIFE ncnn Vulkan, optional but required for 60 fps interpolation

## Install

Install `uv`:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If this is your current PowerShell session immediately after installing `uv`,
run:

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
```

Install FFmpeg with NVIDIA encoder support and add its `bin` directory to PATH
if you are setting this up on another machine. Recommended Windows builds are
from:

```text
https://www.gyan.dev/ffmpeg/builds/
```

After installation, restart PowerShell and check:

```powershell
uv run python upscaler.py --check
```

You should see `h264_nvenc: yes`. `scale_cuda` or `scale_npp` is ideal, but the
script can still use NVENC encoding if only CPU scaling is available.
For AI upscaling, `realesrgan` should show a path instead of `not found`.
For interpolation, `rife` should show a path instead of `not found`.
For enhancement filters, `cas`, `deblock`, and `hqdn3d` should ideally
show `yes`.

Install RIFE for GPU frame interpolation from:

```text
https://github.com/nihui/rife-ncnn-vulkan/releases
```

Extract the Windows zip to:

```text
C:\Tools\rife-ncnn-vulkan
```

Then run `.\setup-path.ps1` or add that folder to PATH.

## UI

```powershell
.\run-ui.ps1
```

From Command Prompt, or by double-clicking:

```bat
run-ui.bat
```

Choose the source video, choose a workflow, choose the output path, and click
`Process video`.

At startup the UI checks which tools are available (FFmpeg, NVENC, Real-ESRGAN,
RIFE) and shows a one-line status below the Workflow selector.

While a job runs all input controls lock and a **Cancel** button replaces the
Process button — clicking it terminates the active FFmpeg or Real-ESRGAN
process. An elapsed-time counter appears next to the progress label. On
successful completion an **Open folder** button reveals the output directory in
Explorer.

Codec, quality, workflow, enhance, and interpolation preferences are saved
automatically when the window closes and restored on next launch.

## CLI

```powershell
uv run python upscaler.py input.mp4
```

That uses the default FFmpeg engine with enhancement on. It applies weak
deblocking, light denoising, and CAS sharpening, then encodes with NVENC.
The output keeps the same resolution as the input.

To skip the enhance filters and just re-encode:

```powershell
uv run python upscaler.py input.mp4 --no-enhance
```

To AI upscale any video to 2x its resolution:

```powershell
uv run python upscaler.py input.mp4 --engine ai
```

That extracts frames, runs Real-ESRGAN at 2x, and reassembles the video. With
`--enhance`, deblocking and denoising are applied as pre-filters before the AI
pass. The output is 2x the input resolution regardless of source size.

To downscale the AI output to a specific resolution:

```powershell
uv run python upscaler.py input.mp4 --engine ai --target 1920x1080
```

To interpolate an existing video to exact `60 fps` without changing its
resolution:

```powershell
uv run python upscaler.py input.mp4 --engine interp60
```

To add final-stage interpolation to any workflow:

```powershell
uv run python upscaler.py input.mp4 --interp60
uv run python upscaler.py input.mp4 --engine ai --interp60
```

Interpolation uses `rife-ncnn-vulkan` on the GPU and targets exact `60 fps`.
Inputs already at `60 fps` are copied or passed through cleanly. Inputs above
`60 fps` fail with a clear error instead of trying to interpolate downward.

The batch launcher also forwards CLI arguments:

```bat
run-ui.bat input.mp4 --engine ai
```

If Windows still has not picked up the PATH changes, run:

```powershell
.\setup-path.ps1
```

To add `uv` and FFmpeg to the system PATH for all users, open PowerShell as
Administrator and run:

```powershell
.\setup-system-path.ps1
```

Useful options:

```powershell
uv run python upscaler.py input.mp4 --codec hevc --quality 18 --overwrite
```

For animation/anime content, try:

```powershell
uv run python upscaler.py input.mp4 --engine ai --model realesr-animevideov3
```

Lower `--quality` values produce larger, higher-quality files. The default is
`19`, which is a good starting point for H.264 NVENC.

## Notes

- The FFmpeg engine preserves the input resolution unless `--target` is given.
- AI mode outputs at 2x the input resolution unless `--target` is given.
- Standalone interpolation preserves the input resolution and converts only the
  frame rate.
- AI mode uses conservative Real-ESRGAN tiling to avoid block/tile corruption.
- 60 fps interpolation uses RIFE on the GPU. It is still frame-heavy at 4K, but
  avoids FFmpeg's slow CPU `minterpolate` path.
- Audio and subtitles are copied when possible.

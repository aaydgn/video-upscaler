# NVIDIA Video Upscaler

Small Python/Tkinter wrapper for improving soft 1080p video with FFmpeg/NVENC
and optional Real-ESRGAN AI workflows. The default workflow is a fast 1080p
enhancement pass.

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
For the fast enhancement mode, `cas`, `deblock`, and `hqdn3d` should ideally
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
RIFE) and shows a one-line status below the Workflow selector. AI workflows that
require a missing tool will warn before the job starts.

While a job runs all input controls lock and a **Cancel** button replaces the
Process button — clicking it terminates the active FFmpeg or Real-ESRGAN
process. An elapsed-time counter appears next to the progress label. On
successful completion an **Open folder** button reveals the output directory in
Explorer.

Before a job starts the UI validates that the input file exists, the output
directory is writable, and the video resolution matches the selected workflow
(for example, 720p AI upscale requires a `1280×720` source). Mismatches show a
warning dialog with the option to proceed or cancel.

Codec, quality, workflow, and interpolation preferences are saved automatically
when the window closes and restored on next launch.

## CLI

```powershell
uv run python upscaler.py "C:\path\to\input.mp4" --output "C:\path\to\output_1080p.mp4"
```

That uses the default fast enhancement workflow. It keeps the final output at
`1920x1080`, applies weak deblocking, light denoising, and CAS sharpening, then
encodes with NVENC.

To interpolate an existing rendered video to exact `60 fps` without changing its
resolution:

```powershell
uv run python upscaler.py input.mp4 --engine interp60
```

For the old 720p AI workflow:

```powershell
uv run python upscaler.py input_720p.mp4 --engine ai
```

That expects the original `1280x720` input, extracts frames, upscales them 2x
with Real-ESRGAN, saves the 2x AI master, then downscales that master to 1080p
with copied audio. The UI and CLI report AI progress using completed frame
counts, so the main progress indicator will not reset when Real-ESRGAN's own
percentage output does.

The batch launcher also forwards CLI arguments:

```bat
run-ui.bat "C:\path\to\input.mp4" --output "C:\path\to\output_1080p.mp4"
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

Use the old fast scaler when you only want a resize:

```powershell
uv run python upscaler.py input.mp4 --engine ffmpeg
```

To add final-stage interpolation to any workflow:

```powershell
uv run python upscaler.py input.mp4 --interp60
uv run python upscaler.py input.mp4 --engine ai4k --interp60
```

Interpolation uses `rife-ncnn-vulkan` on the GPU and targets exact `60 fps`.
Inputs already at `60 fps` are copied or passed through cleanly. Inputs above
`60 fps` fail with a clear error instead of trying to interpolate downward.

For a source that is already 1080p but looks soft, try the fast enhancement
workflow first:

```powershell
uv run python upscaler.py input.mp4 --engine enhance
```

This keeps the final output at `1920x1080`, applies weak deblocking, light
denoising, and CAS sharpening, then encodes with NVENC. It is much faster than
AI 2x from a 1080p source.

The second 1080p workflow is AI re-detailing without going to 4K:

```powershell
uv run python upscaler.py input.mp4 --engine redetail
```

That route expects a `1920x1080` source, denoises/deblocks while downscaling
frames to `960x540`, runs Real-ESRGAN at 2x, and assembles a final
`1920x1080` output. It is slower than fast enhancement but much cheaper than
`1080p -> 4K AI -> 1080p`.

For full 1080p to 4K AI upscaling:

```powershell
uv run python upscaler.py input.mp4 --engine ai4k
```

That route expects a `1920x1080` source and saves a `3840x2160` output.
The 2x AI workflows use `realesr-animevideov3` because the installed
Real-ESRGAN build crops/breaks frames when the x4 models are forced through
`-s 2`.

For animation/anime content, try:

```powershell
uv run python upscaler.py input.mp4 --model realesr-animevideov3
```

Lower `--quality` values produce larger, higher-quality files. The default is
`19`, which is a good starting point for H.264 NVENC.

## Notes

- The target output is fixed at `1920x1080`.
- Standalone interpolation preserves the input resolution and converts only the
  frame rate.
- 720p AI mode saves both a 2x master, for example `<name>_2x.mp4`, and the final
  `1920x1080` output.
- AI modes use conservative Real-ESRGAN tiling to avoid block/tile corruption.
- 60 fps interpolation uses RIFE on the GPU. It is still frame-heavy at 4K, but
  avoids FFmpeg's slow CPU `minterpolate` path.
- Audio and subtitles are copied when possible.
- 720p AI mode expects `1280x720`; AI re-detail mode expects `1920x1080`.

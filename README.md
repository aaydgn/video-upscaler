# NVIDIA Video Upscaler

Small Python/Tkinter wrapper for improving soft 1080p video with FFmpeg/NVENC
and optional Real-ESRGAN AI workflows. The default workflow is a fast 1080p
enhancement pass.

## Requirements

- NVIDIA RTX GPU and current NVIDIA driver
- `uv`
- FFmpeg with NVENC support on `PATH`
- Real-ESRGAN ncnn Vulkan, optional but required for AI upscaling

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
For the fast enhancement mode, `cas`, `deblock`, and `hqdn3d` should ideally
show `yes`.

## UI

```powershell
.\run-ui.ps1
```

From Command Prompt, or by double-clicking:

```bat
run-ui.bat
```

Choose the source video, choose a workflow, choose the output path, and click
`Process to 1080p`.

## CLI

```powershell
uv run python upscaler.py "C:\path\to\input.mp4" --output "C:\path\to\output_1080p.mp4"
```

That uses the default fast enhancement workflow. It keeps the final output at
`1920x1080`, applies weak deblocking, light denoising, and CAS sharpening, then
encodes with NVENC.

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

For animation/anime content, try:

```powershell
uv run python upscaler.py input.mp4 --model realesr-animevideov3
```

Lower `--quality` values produce larger, higher-quality files. The default is
`19`, which is a good starting point for H.264 NVENC.

## Notes

- The target output is fixed at `1920x1080`.
- 720p AI mode saves both a 2x master, for example `<name>_2x.mp4`, and the final
  `1920x1080` output.
- AI modes use conservative Real-ESRGAN tiling to avoid block/tile corruption.
- Audio and subtitles are copied when possible.
- 720p AI mode expects `1280x720`; AI re-detail mode expects `1920x1080`.

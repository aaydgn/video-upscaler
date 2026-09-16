# Video Upscaler — Migration Plan

## Current Architecture

Shell out to `realesrgan-ncnn-vulkan.exe` which reads a folder of extracted
JPEG frames, runs each through the GPU via Vulkan, and writes upscaled frames
to another folder. FFmpeg extracts frames before and reassembles after.

**Bottlenecks:**
- Frames written to disk twice (extract + upscale output)
- ncnn-vulkan runtime is slower than ONNX/TensorRT for the same model
- No temporal awareness — each frame processed independently
- Real-ESRGAN x4plus uses ~5GB VRAM on our 8GB RTX 5060 Ti

## Target Architecture

ONNX Runtime running lightweight models (SPAN/Compact) in-process, with FFmpeg
pipes for decode/encode. No intermediate files on disk.

```
FFmpeg decode (stdout pipe)
    → Python reads raw RGB frames
    → ONNX Runtime upscales on GPU
    → Python writes upscaled frames
    → FFmpeg encode (stdin pipe)
```

## Upscaling Models Landscape (September 2026)

### Comparison Table

| Model | Scale | Video Native? | 8GB OK? | Quality vs x4plus | Speed vs x4plus | Best For |
|---|---|---|---|---|---|---|
| **Real-ESRGAN x4plus** | 4x | No | Yes (~5GB) | Baseline | Baseline | General photos, mixed content |
| **realesr-animevideov3** | 1-4x | No | Yes (~2-3GB) | Lower detail, better temporal | ~3x faster | Anime video |
| **4x-UltraSharp** (community) | 4x | No | Yes (~5GB) | Better on photos | Same | Drop-in upgrade for photos |
| **BSRGAN** | 2x, 4x | No | Yes (~5GB) | Marginally better on degraded | Same | Heavy degradation |
| **SPAN** | 2-4x | No | Yes (<2GB) | Comparable or slightly better | Same or faster | Speed + efficiency |
| **APISR** | 2x, 4x | No | Yes (~1-2GB) | Better (anime only) | Faster | Anime, compressed streaming |
| **Real-CUGAN Pro** | 2-3x | No | Yes (~4-6GB) | Better (anime only) | ~3x slower | Highest anime per-frame quality |
| **HAT** | 2-4x | No | Tight (6-10GB) | +0.5-0.6dB PSNR | 3-5x slower | High-quality single images |
| **SUPIR** | 2-4x | No | Barely (fp8) | Dramatically better (images) | 10-50x slower | Faces, degraded photos |
| **SeedVR2 7B** | 2-4x | **Yes** | Yes (GGUF Q4) | Substantially better | 10-50x slower | Best open-source video quality |
| **FlashVSR** | 2-4x | **Yes** | Unlikely (12GB+) | Much better temporal | Slower (17fps A100) | Cutting-edge video SR |
| **BasicVSR++** | 4x | **Yes** | No (12GB+) | Better temporal (85+ VMAF) | 5-10x slower | Production archive restoration |
| **Video2X 6.4** | 2-4x | **Yes** (framework) | Yes | Same as backend | Same as backend | Best all-in-one pipeline |
| **Anime4K** | 2x | **Yes** (shaders) | Yes (<1GB) | Lower quality | Real-time | Live anime playback |

### Does Anything Beat Real-ESRGAN x4plus?

Yes, but with tradeoffs that matter for our setup (RTX 5060 Ti 8GB):

**Zero-effort upgrades (same speed, same VRAM):**
- **4x-UltraSharp** community model — drop-in weight swap, widely considered
  better on photos
- **BSRGAN** — marginally better on heavily degraded/compressed sources,
  identical architecture

**The real next-gen contender:**
- **SeedVR2** (ByteDance, ICLR 2026) — first open-source model with native
  temporal video processing. 7B model runs on 8GB via GGUF Q4 quantization +
  BlockSwap. Quality genuinely rivals Topaz. But it's 10-50x slower than
  Real-ESRGAN. Reserve it for short clips or precious footage.

**For anime specifically:** x4plus was never the right choice. Use
animevideov3 (3x faster, better temporal stability) or Real-CUGAN Pro via
Video2X (sharper lines, better textures in blind tests). APISR is the dark
horse: only 1.03M parameters (16x smaller than ESRGAN) yet SOTA on anime
benchmarks at CVPR 2024.

**Emerging:** SPAN (NTIRE 2024 champion) matches x4plus quality at equal or
faster speed with <2GB VRAM. Best tooling/ecosystem ratio for our use case.

### Bottom Line

Real-ESRGAN x4plus is no longer technically the best at anything, but nothing
else matches its speed + VRAM + ecosystem combination for general video on
8GB. The field has moved past it technically, but the practical alternatives
haven't matched its ecosystem maturity — yet. SeedVR2 and FlashVSR represent
where things are heading (native temporal diffusion), and within 1-2 years
consumer-GPU-feasible diffusion VSR will likely make per-frame upscaling
obsolete.

## Migration Plan

### Phase 1: ONNX Runtime + FFmpeg Pipe

Replace the ncnn binary with in-process ONNX inference and eliminate disk I/O
for intermediate frames.

**New modules:**
- `upscaler/onnx_upscale.py` — load ONNX model, pre/post-process frames,
  run inference via ONNX Runtime
- `upscaler/pipeline.py` — FFmpeg decode pipe → upscale → FFmpeg encode pipe

**Runtime backend:**
- Try `onnxruntime-gpu` (CUDA) first — if CUDAExecutionProvider is available
  on Blackwell, use it
- Fall back to `onnxruntime-directml` (DirectX 12) — works on any GPU,
  ~20-50% slower than native CUDA but no architecture compatibility issues
- Keep `realesrgan-ncnn-vulkan` as a legacy fallback

**Default models (ship with the app, ~4MB total):**

| Model | Architecture | Scale | Size | Use Case |
|---|---|---|---|---|
| 2xHFA2kSPAN | SPAN | 2x | 1.7MB | Default 2x |
| 4xNomosUni_span_multijpg | SPAN | 4x | 1.7MB | Default 4x |

Additional models (user downloads or we fetch on demand):

| Model | Architecture | Scale | Size |
|---|---|---|---|
| 2xHFA2kCompact | SRVGGNetCompact | 2x | 2.4MB |
| RealESRGAN_x4plus | RRDBNet | 4x | 67MB |
| RealESRGAN_x4plus_anime_6B | RRDBNet | 4x | 18MB |

**ONNX inference pattern:**
```python
# Input:  HWC uint8 RGB numpy array
# Output: HWC uint8 RGB numpy array (scale * H, scale * W)
tensor = frame.astype(np.float32) / 255.0           # normalize
tensor = np.transpose(tensor, (2, 0, 1))             # HWC -> CHW
tensor = np.expand_dims(tensor, axis=0)              # -> NCHW
result = session.run([out_name], {in_name: tensor})[0]
output = np.clip(result.squeeze(0).transpose(1,2,0) * 255, 0, 255).astype(np.uint8)
```

**FFmpeg pipe pattern:**
```
Decoder:  ffmpeg -hwaccel auto -i input.mp4 -f rawvideo -pix_fmt rgb24 -vsync 0 -
Encoder:  ffmpeg -y -f rawvideo -pix_fmt rgb24 -s WxH -r FPS -i pipe:0
          -i input.mp4 -map 0:v -map 1:a? -map 1:s?
          -c:v h264_nvenc -preset p4 -cq 19 -pix_fmt yuv420p
          -c:a copy -c:s copy -shortest -movflags +faststart output.mp4
```

### Phase 2: Model Management

- Scan a `models/` directory for `.onnx` files
- Auto-download default SPAN models on first run
- GUI shows available models with friendly names
- CLI `--model` accepts a path to any ONNX file

### Phase 3: Future Backends (Optional)

- SeedVR2 integration for maximum quality (slow, temporal-aware)
- TensorRT compilation for NVIDIA GPUs (fastest possible inference)
- Community ONNX model marketplace / download UI

## Hardware Target

- NVIDIA GeForce RTX 5060 Ti, 8GB VRAM
- CUDA 13.1, compute capability sm_120 (Blackwell)
- Windows 11
- Official onnxruntime-gpu may not support sm_120 yet — DirectML as fallback

## Dependencies to Add

```
onnxruntime-directml   # ~30MB, works on any GPU
numpy                  # already available in Python stdlib-adjacent
```

Optional:
```
onnxruntime-gpu        # ~200MB, faster on NVIDIA if sm_120 supported
```

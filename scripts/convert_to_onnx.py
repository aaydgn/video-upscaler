"""Convert a PyTorch upscaling model (.pth / .safetensors) to ONNX.

Requires extra dependencies (not part of the main app):
    pip install torch spandrel

Usage:
    python scripts/convert_to_onnx.py path/to/model.pth
    python scripts/convert_to_onnx.py path/to/model.safetensors -o models/my_model.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def convert(input_path: str, output_path: str | None = None, opset: int = 17) -> None:
    try:
        import torch
    except ImportError:
        print("PyTorch is required: pip install torch", file=sys.stderr)
        sys.exit(1)
    try:
        import spandrel
    except ImportError:
        print("spandrel is required: pip install spandrel", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path}...")
    model_desc = spandrel.ModelLoader().load_from_file(input_path)
    model = model_desc.model.eval()
    scale = model_desc.scale
    arch_name = model_desc.architecture.name

    if output_path is None:
        output_path = str(Path(input_path).with_suffix(".onnx"))

    dummy = torch.randn(1, 3, 64, 64)
    print(f"Exporting to ONNX (scale={scale}, arch={arch_name}, opset={opset})...")

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Done: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert upscaling model to ONNX")
    parser.add_argument("input", help="Path to .pth or .safetensors file")
    parser.add_argument("-o", "--output", help="Output .onnx path (default: same name with .onnx extension)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()
    convert(args.input, args.output, args.opset)

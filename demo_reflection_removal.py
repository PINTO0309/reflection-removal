#!/usr/bin/env python3
"""ONNX-based demo for reflection removal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reflection-removal inference using an ONNX model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("dinov3_vitt_gennerator_640x640_640x640.onnx"),
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to an image file or a directory containing images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/demo_outputs"),
        help="Directory where output images will be written.",
    )
    parser.add_argument(
        "--provider",
        dest="providers",
        default="CUDAExecutionProvider",
        help=(
            "Execution provider name for onnxruntime (e.g. CUDAExecutionProvider, "
            "CPUExecutionProvider). Supply multiple times for fallbacks."
        ),
    )
    return parser.parse_args()


def collect_image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if files:
            return files
        raise FileNotFoundError(f"No supported images found in directory '{path}'.")
    raise FileNotFoundError(f"Input path '{path}' does not exist.")


def load_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unable to load image '{path}'.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def convert_to_bgr_uint8(rgb: np.ndarray) -> np.ndarray:
    clipped = np.clip(rgb, 0.0, 1.0)
    bgr = cv2.cvtColor((clipped * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr


def infer_input_size(session: ort.InferenceSession) -> tuple[int, int] | None:
    shape = session.get_inputs()[0].shape
    if len(shape) != 4:
        return None
    height, width = shape[2], shape[3]
    try:
        return int(height), int(width)
    except (TypeError, ValueError):
        return None


def resize_for_model(image: np.ndarray, size_hw: tuple[int, int] | None) -> np.ndarray:
    if size_hw is None:
        return image
    height, width = size_hw
    if image.shape[0] == height and image.shape[1] == width:
        return image
    interpolation = cv2.INTER_AREA if image.shape[0] > height or image.shape[1] > width else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


def prepare_input_tensor(image_rgb: np.ndarray) -> np.ndarray:
    tensor = image_rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.expand_dims(tensor, axis=0)


def postprocess_output(
    output: np.ndarray,
    original_size: tuple[int, int],
) -> np.ndarray:
    data = np.squeeze(output, axis=0)
    data = np.transpose(data, (1, 2, 0))
    if (data.shape[1], data.shape[0]) != original_size:
        data = cv2.resize(data, original_size, interpolation=cv2.INTER_CUBIC)
    return data


def resolve_providers(requested: Sequence[str] | None) -> list[str] | None:
    if not requested:
        return None
    if isinstance(requested, str):
        requested = [item.strip() for item in requested.split(",") if item.strip()]
        if not requested:
            return None
    available = set(ort.get_available_providers())
    providers: list[str] = []
    for name in requested:
        if name not in available:
            raise ValueError(
                f"Execution provider '{name}' is not available. "
                f"Available providers: {sorted(available)}"
            )
        providers.append(name)
    if (
        "CPUExecutionProvider" in available
        and providers[-1] != "CPUExecutionProvider"
    ):
        providers.append("CPUExecutionProvider")
    return providers


def run_inference(
    session: ort.InferenceSession,
    image_path: Path,
    input_name: str,
    model_hw: tuple[int, int] | None,
    output_dir: Path,
) -> None:
    rgb = load_image_rgb(image_path)
    original_hw = (rgb.shape[1], rgb.shape[0])
    resized = resize_for_model(rgb, model_hw)
    tensor = prepare_input_tensor(resized)

    outputs_info = session.get_outputs()
    output_names = [info.name for info in outputs_info]
    raw_outputs = session.run(output_names, {input_name: tensor})
    outputs = dict(zip(output_names, raw_outputs))

    transmission = outputs.get("transmission")
    if transmission is None:
        transmission = raw_outputs[0]
    reflection = outputs.get("reflection")
    if reflection is None and len(raw_outputs) > 1:
        reflection = raw_outputs[1]

    transmission_rgb = postprocess_output(transmission, original_hw)
    transmission_path = output_dir / f"{image_path.stem}_transmission.png"
    if not cv2.imwrite(str(transmission_path), convert_to_bgr_uint8(transmission_rgb)):
        raise RuntimeError(f"Failed to write '{transmission_path}'.")

    if reflection is not None:
        reflection_rgb = postprocess_output(reflection, original_hw)
        reflection_path = output_dir / f"{image_path.stem}_reflection.png"
        if not cv2.imwrite(str(reflection_path), convert_to_bgr_uint8(reflection_rgb)):
            raise RuntimeError(f"Failed to write '{reflection_path}'.")


def main() -> int:
    args = parse_args()
    try:
        providers = resolve_providers(args.providers)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    model_path = args.model.expanduser()
    if not model_path.is_file():
        print(f"error: ONNX model '{model_path}' not found.", file=sys.stderr)
        return 1

    output_dir = args.output.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        image_paths = collect_image_paths(args.input.expanduser())
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    session = ort.InferenceSession(
        str(model_path),
        providers=providers or None,
    )
    model_hw = infer_input_size(session)
    input_name = session.get_inputs()[0].name

    for path in image_paths:
        try:
            run_inference(session, path, input_name, model_hw, output_dir)
            print(f"Inference complete: {path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"error processing '{path}': {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

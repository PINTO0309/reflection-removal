#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from models import DINOFeatureExtractor, HypercolumnGenerator, VGGFeatureExtractor

BACKBONE_CHOICES = ["vgg19", "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the reflection removal generator to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to generator checkpoint (.pt).")
    parser.add_argument("--output", type=Path, default=Path("generator.onnx"), help="ONNX export path.")
    parser.add_argument("--backbone", choices=BACKBONE_CHOICES, help="Backbone used during training.")
    parser.add_argument("--ckpt_dir", type=Path, default=Path("ckpts"), help="Directory containing optional backbone weights.")
    parser.add_argument("--device", default="cpu", help="Device for model initialisation (e.g. cpu, cuda:0).")
    parser.add_argument("--use_hyper", type=int, choices=[0, 1], default=1, help="Whether hypercolumn inputs were enabled during training.")
    parser.add_argument("--batch_size", type=int, default=1, help="Dummy batch size for export.")
    parser.add_argument("--height", type=int, default=512, help="Dummy input height.")
    parser.add_argument("--width", type=int, default=512, help="Dummy input width.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--static_shape", action="store_true", help="Disable dynamic height/width axes in the ONNX graph.")
    return parser.parse_args()


def _normalise_backbone(name: str) -> str:
    candidate = name.lower()
    if candidate not in BACKBONE_CHOICES:
        raise ValueError(f"Unsupported backbone '{name}'. Expected one of {BACKBONE_CHOICES}.")
    return candidate


def resolve_backbone(state: Any, override: str | None) -> str:
    if override:
        override_norm = _normalise_backbone(override)
    else:
        override_norm = None
    if isinstance(state, dict):
        meta = state.get("backbone")
        if isinstance(meta, str):
            meta_norm = _normalise_backbone(meta)
            if override_norm and override_norm != meta_norm:
                raise ValueError(f"Checkpoint backbone '{meta_norm}' does not match '--backbone {override_norm}'.")
            return meta_norm
    if override_norm:
        return override_norm
    raise ValueError("Backbone not provided and checkpoint does not store the backbone identifier.")


def create_feature_extractor(backbone: str, use_hyper: bool, ckpt_dir: Path):
    if backbone == "vgg19":
        return VGGFeatureExtractor(use_hyper=use_hyper)
    if backbone in DINOFeatureExtractor.CKPT_FILENAMES:
        return DINOFeatureExtractor(backbone, use_hyper=use_hyper, ckpt_root=ckpt_dir)
    raise ValueError(f"Unsupported backbone '{backbone}'.")


def load_generator_state(generator: HypercolumnGenerator, state: Any, backbone: str) -> None:
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            ckpt_backbone = state.get("backbone")
            if isinstance(ckpt_backbone, str):
                ckpt_norm = _normalise_backbone(ckpt_backbone)
                if ckpt_norm != backbone:
                    raise ValueError(f"Checkpoint backbone '{ckpt_norm}' does not match requested backbone '{backbone}'.")
            generator.load_state_dict(state["state_dict"])
            return
        if "generator" in state and isinstance(state["generator"], dict):
            ckpt_backbone = state.get("backbone")
            if isinstance(ckpt_backbone, str):
                ckpt_norm = _normalise_backbone(ckpt_backbone)
                if ckpt_norm != backbone:
                    raise ValueError(f"Checkpoint backbone '{ckpt_norm}' does not match requested backbone '{backbone}'.")
            generator.load_state_dict(state["generator"])
            return
    generator.load_state_dict(state)


def build_dummy_input(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    batch_size: int = args.batch_size
    height: int = args.height
    width: int = args.width
    shape = (batch_size, 3, height, width)
    return torch.randn(shape, device=device)


def export_onnx(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint: Path = args.checkpoint
    backbone: str = args.backbone
    checkpoint_path = checkpoint.expanduser()
    state = torch.load(checkpoint_path, map_location=device)
    backbone = resolve_backbone(state, backbone)
    use_hyper = bool(args.use_hyper)
    output: Path = args.output
    ckpt_dir: Path = args.ckpt_dir
    static_shape: bool = args.static_shape

    feature_extractor = create_feature_extractor(backbone, use_hyper, ckpt_dir)
    feature_extractor.eval()
    generator = HypercolumnGenerator(feature_extractor).to(device)
    generator.eval()
    load_generator_state(generator, state, backbone)

    dummy_input = build_dummy_input(args, device)

    dynamic_axes: Dict[str, Dict[int, str]] | None = None
    if not static_shape:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "transmission": {0: "batch", 2: "height", 3: "width"},
            "reflection": {0: "batch", 2: "height", 3: "width"},
        }

    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        generator,
        (dummy_input,),
        output,
        input_names=["input"],
        output_names=["transmission", "reflection"],
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
    )
    print(f"[i] Exported ONNX model to {output.resolve()}")


def main() -> None:
    torch.set_grad_enabled(False)
    args = parse_args()
    export_onnx(args)


if __name__ == "__main__":
    main()

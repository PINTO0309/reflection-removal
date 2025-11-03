#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

import onnx
from onnxslim import slim
from onnxsim import simplify

from models import (
    DINOFeatureExtractor,
    HGNetFeatureExtractor,
    HypercolumnGenerator,
    VGGFeatureExtractor,
)

BACKBONE_CHOICES = [
    "vgg19",
    "hgnetv2",
    "dinov3_vitt",
    "dinov3_vits16",
    "dinov3_vits16plus",
    "dinov3_vitb16",
]

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
    parser.add_argument(
        "--head_only",
        action="store_true",
        help="Export only the generator head. Useful when backbone features are provided externally.",
    )
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
    if backbone == "hgnetv2":
        return HGNetFeatureExtractor(use_hyper=use_hyper, ckpt_root=ckpt_dir)
    if backbone in DINOFeatureExtractor.CKPT_FILENAMES:
        return DINOFeatureExtractor(backbone, use_hyper=use_hyper, ckpt_root=ckpt_dir)
    raise ValueError(f"Unsupported backbone '{backbone}'.")


class GeneratorExportWrapper(torch.nn.Module):
    def __init__(self, generator: HypercolumnGenerator):
        super().__init__()
        self.generator = generator

    def forward(self, x: torch.Tensor):
        transmission, reflection = self.generator(x)
        return transmission, reflection


class GeneratorHeadExportWrapper(torch.nn.Module):
    def __init__(self, generator: HypercolumnGenerator):
        super().__init__()
        self.generator = generator

    @property
    def input_channels(self) -> int:
        return self.generator.conv0.conv.in_channels

    def forward(self, hyper_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        net = self.generator.conv0(hyper_input)
        net = self.generator.conv1(net)
        net = self.generator.conv2(net)
        net = self.generator.conv3(net)
        net = self.generator.conv4(net)
        net = self.generator.conv5(net)
        net = self.generator.conv6(net)
        net = self.generator.conv7(net)
        net = self.generator.conv9(net)
        outputs = self.generator.conv_last(net)
        transmission, reflection = torch.chunk(outputs, 2, dim=1)
        return transmission, reflection


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


def build_dummy_input(args: argparse.Namespace, device: torch.device, channels: int = 3) -> torch.Tensor:
    batch_size: int = args.batch_size
    height: int = args.height
    width: int = args.width
    shape = (batch_size, channels, height, width)
    return torch.randn(shape, device=device)


def simplify_onnx(output: Path, dynamic: bool) -> None:
    model = onnx.load(str(output))
    slimmed_model = slim(model)
    simplified_model, check = simplify(slimmed_model, dynamic_input_shape=dynamic)
    if not check:
        raise RuntimeError("ONNX simplification check failed; generated model may be invalid.")
    onnx.save(simplified_model, str(output))
    print(f"[i] Simplified ONNX model saved to {output.resolve()}")
    decompose_batch_normalization(output, dynamic)


def decompose_batch_normalization(output: Path, dynamic: bool) -> None:
    model = onnx.load(str(output))
    graph = model.graph
    existing_initializers = {init.name for init in graph.initializer}
    nodes = list(graph.node)
    graph.ClearField("node")

    axes = [0, 2, 3]

    for node in nodes:
        if node.op_type != "BatchNormalization":
            graph.node.append(node)
            continue

        x, scale, bias, mean, var = node.input[:5]
        epsilon = next((attr.f for attr in node.attribute if attr.name == "epsilon"), 1e-5)
        out = node.output[0]

        base_name = node.name or out
        eps_name = f"{base_name}_eps"
        suffix = 0
        while eps_name in existing_initializers:
            suffix += 1
            eps_name = f"{base_name}_eps_{suffix}"
        existing_initializers.add(eps_name)

        eps_const = onnx.helper.make_tensor(
            name=eps_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=[],
            vals=[epsilon],
        )
        graph.initializer.append(eps_const)

        axes_name = f"{base_name}_unsqueeze_axes"
        suffix = 0
        while axes_name in existing_initializers:
            suffix += 1
            axes_name = f"{base_name}_unsqueeze_axes_{suffix}"
        existing_initializers.add(axes_name)

        axes_tensor = onnx.helper.make_tensor(
            name=axes_name,
            data_type=onnx.TensorProto.INT64,
            dims=[len(axes)],
            vals=axes,
        )
        graph.initializer.append(axes_tensor)

        add_node = onnx.helper.make_node(
            "Add",
            inputs=[var, eps_const.name],
            outputs=[f"{base_name}_var_eps"],
            name=f"{base_name}_add",
        )
        sqrt_node = onnx.helper.make_node(
            "Sqrt",
            inputs=[add_node.output[0]],
            outputs=[f"{base_name}_sqrt"],
            name=f"{base_name}_sqrt",
        )
        div_scale_node = onnx.helper.make_node(
            "Div",
            inputs=[scale, sqrt_node.output[0]],
            outputs=[f"{base_name}_scale"],
            name=f"{base_name}_div_scale",
        )
        div_scale_unsq_node = onnx.helper.make_node(
            "Unsqueeze",
            inputs=[div_scale_node.output[0], axes_name],
            outputs=[f"{base_name}_scale_unsq"],
            name=f"{base_name}_unsqueeze_scale",
        )
        mean_unsq_node = onnx.helper.make_node(
            "Unsqueeze",
            inputs=[mean, axes_name],
            outputs=[f"{base_name}_mean_unsq"],
            name=f"{base_name}_unsqueeze_mean",
        )
        sub_mean_node = onnx.helper.make_node(
            "Sub",
            inputs=[x, mean_unsq_node.output[0]],
            outputs=[f"{base_name}_centered"],
            name=f"{base_name}_sub_mean",
        )
        mul_node = onnx.helper.make_node(
            "Mul",
            inputs=[sub_mean_node.output[0], div_scale_unsq_node.output[0]],
            outputs=[f"{base_name}_normalized"],
            name=f"{base_name}_mul",
        )
        bias_unsq_node = onnx.helper.make_node(
            "Unsqueeze",
            inputs=[bias, axes_name],
            outputs=[f"{base_name}_bias_unsq"],
            name=f"{base_name}_unsqueeze_bias",
        )
        add_bias_node = onnx.helper.make_node(
            "Add",
            inputs=[mul_node.output[0], bias_unsq_node.output[0]],
            outputs=[out],
            name=f"{base_name}_add_bias",
        )

        graph.node.extend(
            [
                add_node,
                sqrt_node,
                div_scale_node,
                div_scale_unsq_node,
                mean_unsq_node,
                sub_mean_node,
                mul_node,
                bias_unsq_node,
                add_bias_node,
            ]
        )

    onnx.save(model, str(output))
    print(f"[i] Decomposed remaining BatchNormalization ops in {output.resolve()}")
    resimplify_after_bn(output, dynamic)


def resimplify_after_bn(output: Path, dynamic: bool) -> None:
    model = onnx.load(str(output))
    resimplified_model, check = simplify(
        model,
        dynamic_input_shape=dynamic,
        skip_fuse_bn=True,
    )
    if not check:
        raise RuntimeError("ONNX simplification after batch norm decomposition failed.")
    onnx.save(resimplified_model, str(output))
    print(f"[i] Re-simplified ONNX model saved to {output.resolve()}")


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

    if args.head_only:
        export_module = GeneratorHeadExportWrapper(generator).to(device)
        export_module.eval()
        input_name = "hyper_input"
        output_names = ["transmission", "reflection"]
        input_channels = export_module.input_channels
    else:
        export_module = GeneratorExportWrapper(generator).to(device)
        export_module.eval()
        input_name = "input"
        output_names = ["transmission", "reflection"]
        input_channels = 3

    dummy_input = build_dummy_input(args, device, channels=input_channels)

    dynamic_axes: Dict[str, Dict[int, str]] | None = None
    if not static_shape:
        dynamic_axes = {
            input_name: {0: "batch", 2: "height", 3: "width"},
            "transmission": {0: "batch", 2: "height", 3: "width"},
            "reflection": {0: "batch", 2: "height", 3: "width"},
        }
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_module,
        (dummy_input,),
        output,
        input_names=[input_name],
        output_names=output_names,
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
    )
    print(f"[i] Exported ONNX model to {output.resolve()}")
    simplify_onnx(output, dynamic=not static_shape)


def main() -> None:
    torch.set_grad_enabled(False)
    args = parse_args()
    export_onnx(args)


if __name__ == "__main__":
    main()

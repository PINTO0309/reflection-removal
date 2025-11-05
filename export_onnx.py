#!/usr/bin/env python3

import contextlib
import os
import re
import tempfile
import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

import onnx
from onnxslim import slim
from onnxsim import simplify

from models import (
    DINOFeatureExtractor,
    HGNetFeatureExtractor,
    HypercolumnGenerator,
    ResidualHypercolumnGenerator,
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
    parser.add_argument(
        "--head_height",
        type=int,
        help="Target head resolution height when exporting with --head_only. Defaults to --height.",
    )
    parser.add_argument(
        "--head_width",
        type=int,
        help="Target head resolution width when exporting with --head_only. Defaults to --width.",
    )
    return parser.parse_args()


def _sanitize_feature_input_name(name: str) -> str:
    candidate = name.lstrip("/")
    if candidate.startswith("generator/"):
        candidate = "backbone-" + candidate[len("generator/") :]
    elif not candidate.startswith("backbone-"):
        candidate = f"backbone-{candidate}"
    candidate = candidate.replace("/", "-")
    return candidate


def _normalise_backbone(name: str) -> str:
    candidate = name.lower()
    if candidate not in BACKBONE_CHOICES:
        raise ValueError(f"Unsupported backbone '{name}'. Expected one of {BACKBONE_CHOICES}.")
    return candidate


GENERATOR_VARIANT_BASELINE = "baseline"
GENERATOR_VARIANT_RESIDUAL = "residual_skips"


def infer_generator_variant(state_dict: Dict[str, Any]) -> str:
    for key in state_dict.keys():
        if key.startswith("residual_scales"):
            return GENERATOR_VARIANT_RESIDUAL
    return GENERATOR_VARIANT_BASELINE


def extract_generator_state(
    state: Any,
    backbone: str,
) -> tuple[Dict[str, torch.Tensor], str]:
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            ckpt_backbone = state.get("backbone")
            if isinstance(ckpt_backbone, str):
                ckpt_norm = _normalise_backbone(ckpt_backbone)
                if ckpt_norm != backbone:
                    raise ValueError(
                        f"Checkpoint backbone '{ckpt_norm}' does not match requested backbone '{backbone}'."
                    )
            state_dict = state["state_dict"]
            variant = state.get("generator_variant")
            if isinstance(variant, str):
                return state_dict, variant
            return state_dict, infer_generator_variant(state_dict)
        if "generator" in state and isinstance(state["generator"], dict):
            ckpt_backbone = state.get("backbone")
            if isinstance(ckpt_backbone, str):
                ckpt_norm = _normalise_backbone(ckpt_backbone)
                if ckpt_norm != backbone:
                    raise ValueError(
                        f"Checkpoint backbone '{ckpt_norm}' does not match requested backbone '{backbone}'."
                    )
            state_dict = state["generator"]
            variant = state.get("generator_variant")
            if isinstance(variant, str):
                return state_dict, variant
            return state_dict, infer_generator_variant(state_dict)
        state_dict = state
        variant = infer_generator_variant(state_dict)
        return state_dict, variant
    raise ValueError("Unsupported checkpoint format for generator weights.")


def state_dict_has_output_skip(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("output_skip_scale") for key in state_dict.keys())


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


def create_feature_extractor(
    backbone: str,
    use_hyper: bool,
    ckpt_dir: Path,
    distributed_hypercolumn: bool = False,
):
    if backbone == "vgg19":
        return VGGFeatureExtractor(use_hyper=use_hyper, distributed_hypercolumn=distributed_hypercolumn)
    if backbone == "hgnetv2":
        return HGNetFeatureExtractor(
            use_hyper=use_hyper,
            ckpt_root=ckpt_dir,
            distributed_hypercolumn=distributed_hypercolumn,
        )
    if backbone in DINOFeatureExtractor.CKPT_FILENAMES:
        return DINOFeatureExtractor(
            backbone,
            use_hyper=use_hyper,
            ckpt_root=ckpt_dir,
            distributed_hypercolumn=distributed_hypercolumn,
        )
    raise ValueError(f"Unsupported backbone '{backbone}'.")


class GeneratorExportWrapper(torch.nn.Module):
    def __init__(self, generator: HypercolumnGenerator, target_size: tuple[int, int] | None = None):
        super().__init__()
        self.generator = generator
        self.target_size = target_size

    def forward(self, x: torch.Tensor):
        transmission, reflection = self.generator(x)
        if self.target_size:
            transmission = F.interpolate(transmission, size=self.target_size, mode="bilinear", align_corners=False)
            reflection = F.interpolate(reflection, size=self.target_size, mode="bilinear", align_corners=False)
        return transmission, reflection


class GeneratorHeadExportWrapper(torch.nn.Module):
    def __init__(
        self,
        generator: HypercolumnGenerator,
        hyper_layers: List[str],
        use_hyper: bool,
        input_names: List[str] | None = None,
        backbone_height: int | None = None,
        backbone_width: int | None = None,
        head_height: int | None = None,
        head_width: int | None = None,
        patch_size: int | None = None,
    ):
        super().__init__()
        self.generator = generator
        self.hyper_layers = list(hyper_layers)
        self.use_hyper = use_hyper
        if input_names is None:
            input_names = [*self.hyper_layers, "input"]
        self._input_names = list(input_names)
        self.backbone_height = backbone_height or head_height or 0
        self.backbone_width = backbone_width or head_width or 0
        self.head_height = head_height or self.backbone_height
        self.head_width = head_width or self.backbone_width
        self.patch_size = patch_size

    @property
    def input_names(self) -> List[str]:
        return self._input_names

    def _apply_concat_preprocess(self, feature: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        target_h, target_w = target_size
        if feature.dim() == 3:
            if not self.patch_size:
                raise ValueError("Token inputs require known patch size.")
            patch = self.patch_size
            batch, tokens_len, channels = feature.shape
            grid_h = self.backbone_height // patch
            grid_w = self.backbone_width // patch
            expected_tokens = grid_h * grid_w
            if tokens_len < expected_tokens + 1:
                raise ValueError("Insufficient tokens provided for backbone resolution.")
            feature = torch.narrow(feature, 1, 1, tokens_len - 1)
            feature = torch.narrow(feature, 1, 0, expected_tokens)
            feature = feature.transpose(1, 2).contiguous()
            feature = feature.reshape(batch, channels, grid_h, grid_w)
        elif feature.dim() == 4:
            pass
        else:
            raise ValueError("Unsupported feature shape for head export.")

        feature = F.interpolate(feature, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return feature

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = len(self.hyper_layers) + 1
        if len(inputs) != expected:
            raise ValueError(f"Expected {expected} inputs (features + image), got {len(inputs)}.")
        *feature_maps, image = inputs
        target_size = (self.head_height, self.head_width)
        prepared_features: List[torch.Tensor] = []
        for feature in feature_maps:
            feature = self._apply_concat_preprocess(feature, target_size)
            if not self.use_hyper:
                feature = torch.zeros_like(feature)
            prepared_features.append(feature)

        image_prepared = F.interpolate(image, size=target_size, mode="bilinear", align_corners=False)

        if prepared_features:
            hyper_input = torch.cat(prepared_features + [image_prepared], dim=1)
        else:
            hyper_input = image_prepared

        outputs = self.generator.forward_head(hyper_input)
        transmission, reflection = torch.chunk(outputs, 2, dim=1)
        skip_scale = getattr(self.generator, "output_skip_scale", None)
        if skip_scale is not None:
            transmission = image_prepared + skip_scale * transmission
        return transmission, reflection


class FullGeneratorExportWrapper(torch.nn.Module):
    def __init__(
        self,
        generator: HypercolumnGenerator,
        head_height: int,
        head_width: int,
    ):
        super().__init__()
        self.generator = generator
        self.feature_extractor = generator.feature_extractor
        self.head_height = head_height
        self.head_width = head_width

    def _resize_to_head(self, tensor: torch.Tensor) -> torch.Tensor:
        size = (self.head_height, self.head_width)
        if tensor.shape[-2:] != size:
            tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
        return tensor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_size = (self.head_height, self.head_width)
        hyper_input, _ = self.feature_extractor.build_hypercolumns_with_features(
            x,
            require_grad=True,
            target_size=target_size,
        )
        image_resized = self._resize_to_head(x)
        outputs = self.generator.forward_head(hyper_input)
        transmission, reflection = torch.chunk(outputs, 2, dim=1)
        skip_scale = getattr(self.generator, "output_skip_scale", None)
        if skip_scale is not None:
            transmission = image_resized + skip_scale * transmission
        return transmission, reflection


def build_dummy_input(
    args: argparse.Namespace,
    device: torch.device,
    channels: int = 3,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    batch_size: int = args.batch_size
    height: int = args.height
    width: int = args.width
    shape = (batch_size, channels, height, width)
    return torch.randn(shape, device=device, dtype=dtype)


def build_head_only_dummy_inputs(
    args: argparse.Namespace,
    device: torch.device,
    feature_keys: List[str],
    layer_dims: Any,
    dtype: torch.dtype,
    patch_size: int | None,
) -> tuple[torch.Tensor, ...]:
    if isinstance(layer_dims, dict):
        dims_map = layer_dims
    else:
        dims_map = dict(layer_dims)
    batch_size: int = args.batch_size
    feature_height: int = args.height
    feature_width: int = args.width
    head_height: int = getattr(args, "head_height", args.head_height)
    head_width: int = getattr(args, "head_width", args.head_width)
    dummy_inputs: List[torch.Tensor] = []
    for name in feature_keys:
        channels = dims_map.get(name)
        if channels is None:
            raise KeyError(f"Layer '{name}' not found in feature extractor dimensions.")
        if patch_size:
            pad_h = (patch_size - feature_height % patch_size) % patch_size
            pad_w = (patch_size - feature_width % patch_size) % patch_size
            grid_h = (feature_height + pad_h) // patch_size
            grid_w = (feature_width + pad_w) // patch_size
            tokens = grid_h * grid_w + 1
            tensor = torch.randn((batch_size, tokens, channels), device=device, dtype=dtype)
        else:
            tensor = torch.randn((batch_size, channels, feature_height, feature_width), device=device, dtype=dtype)
        dummy_inputs.append(tensor)
    rgb = torch.randn((batch_size, 3, args.height, args.width), device=device, dtype=dtype)
    dummy_inputs.append(rgb)
    return tuple(dummy_inputs)


def infer_head_input_names(
    generator: HypercolumnGenerator,
    hyper_layers: List[str],
    args: argparse.Namespace,
    device: torch.device,
    static_shape: bool,
    dtype: torch.dtype,
) -> tuple[List[str], List[str]] | None:
    module = GeneratorExportWrapper(generator).to(device)
    module.eval()
    dummy = build_dummy_input(args, device, channels=3, dtype=dtype)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    output_names = ["transmission", "reflection"]
    dynamic_axes: Dict[str, Dict[int, str]] | None = None
    if not static_shape:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "transmission": {0: "batch", 2: "height", 3: "width"},
            "reflection": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        module,
        (dummy,),
        tmp_path,
        input_names=["input"],
        output_names=output_names,
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
    )

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        simplify_onnx(tmp_path, dynamic=not static_shape)

    try:
        model = onnx.load(str(tmp_path))
        concat_inputs: List[str] | None = None
        for node in model.graph.node:
            if node.op_type == "Concat" and "input" in node.input:
                concat_inputs = list(node.input)
                break
        if concat_inputs is None:
            return None

        pass_through_ops = {"Resize", "Reshape", "Transpose", "Slice", "Identity"}
        feature_sources: List[str] = []
        for name in concat_inputs:
            if name == "input":
                continue
            source = name
            visited: set[str] = set()
            while True:
                if source in visited:
                    break
                visited.add(source)
                producer = next((node for node in model.graph.node if source in node.output), None)
                if producer is None:
                    break
                if producer.op_type in pass_through_ops and producer.input:
                    source = producer.input[0]
                    continue
                break
            feature_sources.append(source)

        sanitized_features = [_sanitize_feature_input_name(name) for name in feature_sources]
        sanitized_inputs = sanitized_features + ["input"]
        return feature_sources, sanitized_inputs
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def enforce_input_names(output: Path, desired_names: List[str]) -> None:
    model = onnx.load(str(output))
    inputs = list(model.graph.input)
    if len(inputs) != len(desired_names):
        return
    mapping: Dict[str, str] = {}
    for value_info, new_name in zip(inputs, desired_names):
        old_name = value_info.name
        if old_name == new_name:
            continue
        mapping[old_name] = new_name
        value_info.name = new_name
    if not mapping:
        return
    for node in model.graph.node:
        node.input[:] = [mapping.get(name, name) for name in node.input]
    for value_info in model.graph.value_info:
        if value_info.name in mapping:
            value_info.name = mapping[value_info.name]
    for output in model.graph.output:
        if output.name in mapping:
            output.name = mapping[output.name]
    for initializer in model.graph.initializer:
        if initializer.name in mapping:
            initializer.name = mapping[initializer.name]
    onnx.save(model, str(output))


def remove_redundant_resize_chains(output: Path) -> None:
    model = onnx.load(str(output))
    output_usage: Dict[str, int] = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                output_usage[name] = output_usage.get(name, 0) + 1

    producers: Dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                producers[out] = node

    to_remove_outputs: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        data_input = node.input[0]
        parent = producers.get(data_input)
        if not parent or parent.op_type != "Resize":
            continue
        parent_output = parent.output[0]
        if output_usage.get(parent_output, 0) != 1:
            continue
        node.input[0] = parent.input[0]
        to_remove_outputs.add(parent_output)

    if not to_remove_outputs:
        return

    nodes_kept = [node for node in model.graph.node if not any(out in to_remove_outputs for out in node.output)]
    model.graph.ClearField("node")
    model.graph.node.extend(nodes_kept)

    value_info_kept = [vi for vi in model.graph.value_info if vi.name not in to_remove_outputs]
    model.graph.ClearField("value_info")
    model.graph.value_info.extend(value_info_kept)
    onnx.save(model, str(output))


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

    state_dict, variant = extract_generator_state(state, backbone)
    has_residual = variant == GENERATOR_VARIANT_RESIDUAL
    has_output_skip = state_dict_has_output_skip(state_dict)

    distributed_hypercolumn = any(
        key.startswith("feature_extractor._distributed_reduction_layers")
        or key.startswith("feature_extractor._distributed_post")
        for key in state_dict.keys()
    )
    feature_extractor = create_feature_extractor(
        backbone,
        use_hyper,
        ckpt_dir,
        distributed_hypercolumn=distributed_hypercolumn,
    )
    feature_extractor.eval()
    if has_residual:
        output_skip_init = 0.0 if has_output_skip else None
        generator = ResidualHypercolumnGenerator(
            feature_extractor,
            residual_init=0.0,
            output_skip_init=output_skip_init,
        ).to(device)
    else:
        generator = HypercolumnGenerator(feature_extractor).to(device)
    generator.eval()
    try:
        generator.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Failed to load generator weights from checkpoint. "
            "Ensure the checkpoint matches the expected architecture."
        ) from error
    param_dtype = next(generator.parameters()).dtype
    head_height = getattr(args, "head_height", None)
    head_width = getattr(args, "head_width", None)
    if head_height is None:
        head_height = args.height
        args.head_height = head_height
    if head_width is None:
        head_width = args.width
        args.head_width = head_width

    if args.head_only:
        hyper_layers = list(feature_extractor.hyper_layers)
        inferred = infer_head_input_names(generator, hyper_layers, args, device, static_shape, param_dtype)
        if inferred is None:
            feature_sources = hyper_layers[:]
            sanitized_inputs = [_sanitize_feature_input_name(name) for name in feature_sources] + ["input"]
        else:
            feature_sources, sanitized_inputs = inferred

        feature_keys: List[str] = []
        remaining_layers = hyper_layers[:]
        for source_name in feature_sources:
            match = next((layer for layer in remaining_layers if layer and layer in source_name), None)
            if match is None:
                digits_source = re.findall(r"\d+", source_name)
                match = next(
                    (
                        layer
                        for layer in remaining_layers
                        if digits_source
                        and re.findall(r"\d+", layer) == digits_source
                    ),
                    None,
                )
            if match is None and remaining_layers:
                match = remaining_layers[0]
            if match is not None and match in remaining_layers:
                remaining_layers.remove(match)
            feature_keys.append(match if match is not None else hyper_layers[0])

        input_names = list(sanitized_inputs)

        patch_size = getattr(feature_extractor, "patch_size", None)

        export_module = GeneratorHeadExportWrapper(
            generator,
            feature_keys,
            use_hyper,
            input_names,
            backbone_height=args.height,
            backbone_width=args.width,
            head_height=head_height,
            head_width=head_width,
            patch_size=patch_size,
        ).to(device)
        export_module.eval()
        output_names = ["transmission", "reflection"]
        dummy_inputs = build_head_only_dummy_inputs(
            args,
            device,
            feature_keys,
            feature_extractor.layer_dims,
            param_dtype,
            patch_size,
        )
    else:
        if head_height != args.height or head_width != args.width:
            export_module = FullGeneratorExportWrapper(generator, head_height, head_width).to(device)
        else:
            export_module = GeneratorExportWrapper(generator).to(device)
        export_module.eval()
        input_names = ["input"]
        output_names = ["transmission", "reflection"]

    if not args.head_only:
        dummy_inputs = (build_dummy_input(args, device, channels=3, dtype=param_dtype),)

    dynamic_axes: Dict[str, Dict[int, str]] | None = None
    if not static_shape:
        if args.head_only:
            dynamic_axes = {}
            for name in input_names:
                if name == "input":
                    dynamic_axes[name] = {0: "batch", 2: "height", 3: "width"}
                else:
                    dynamic_axes[name] = {0: "batch", 1: "tokens"}
        else:
            dynamic_axes = {name: {0: "batch", 2: "height", 3: "width"} for name in input_names}
        dynamic_axes.update(
            {
                "transmission": {0: "batch", 2: "height", 3: "width"},
                "reflection": {0: "batch", 2: "height", 3: "width"},
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_module,
        dummy_inputs,
        output,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
    )
    print(f"[i] Exported ONNX model to {output.resolve()}")
    remove_redundant_resize_chains(output)
    simplify_onnx(output, dynamic=not static_shape)
    remove_redundant_resize_chains(output)
    if args.head_only:
        enforce_input_names(output, input_names)


def main() -> None:
    torch.set_grad_enabled(False)
    args = parse_args()
    export_onnx(args)


if __name__ == "__main__":
    main()

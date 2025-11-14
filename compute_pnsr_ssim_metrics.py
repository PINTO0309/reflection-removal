#!/usr/bin/env python3
"""
Compute PSNR and SSIM statistics for the reflection-dataset pairs.

By default the script evaluates both the real and synthetic splits found under
`reflection-dataset/`:
  * real/blended vs real/transmission_layer
  * synthetic/reflection_layer vs synthetic/transmission_layer

Use `--subset real` or `--subset synthetic` to restrict the evaluation, and
`--output-csv metrics.csv` to persist per-image results for downstream analysis.
Pass `--checkpoint <path>` or `--onnx-model <path>` to benchmark generator
predictions (transmission outputs) against the ground-truth transmission layer.
"""
from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from scipy.stats import norm
from models import (
    DINOFeatureExtractor,
    FeatureExtractorBase,
    HGNetFeatureExtractor,
    HypercolumnGenerator,
    ResidualHypercolumnGenerator,
    ResidualInResidualHypercolumnGenerator,
    VGGFeatureExtractor,
)
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from tqdm import tqdm


PAIR_CONFIG = {
    "real": ("blended", "transmission_layer"),
    "synthetic": ("reflection_layer", "transmission_layer"),
}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BACKBONE_CHOICES = (
    "vgg19",
    "hgnetv2",
    "dinov3_vitt",
    "dinov3_vits16",
    "dinov3_vits16plus",
    "dinov3_vitb16",
)
SYNTHETIC_MIN_WIDTH = 256
SYNTHETIC_MAX_WIDTH = 640
SYNTHETIC_WIDTH_STEP = 16
SYNTHETIC_SIGMAS = np.linspace(1.0, 5.0, 80).tolist()
GAUSSIAN_MASK: Optional[np.ndarray] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PSNR and SSIM for the reflection-dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("reflection-dataset"),
        help="Root folder that contains the 'real' and 'synthetic' splits.",
    )
    parser.add_argument(
        "--subset",
        choices=("real", "synthetic", "all"),
        default="all",
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to store per-image metrics as CSV.",
    )
    parser.add_argument(
        "--skip-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip samples without a matching counterpart instead of raising.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the number of pairs evaluated per split.",
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to a PyTorch generator checkpoint (.pt) or a directory containing generator.pt/checkpoint files.",
    )
    model_group.add_argument(
        "--onnx-model",
        type=Path,
        help="Path to an ONNX generator used for inference.",
    )
    parser.add_argument(
        "--backbone",
        choices=BACKBONE_CHOICES,
        help="Backbone identifier required when loading PyTorch checkpoints (inferred from metadata when omitted).",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("ckpts"),
        help="Directory that stores pretrained backbone weights (used when loading checkpoints).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for checkpoint inference (defaults to CUDA when available).",
    )
    parser.add_argument(
        "--providers",
        choices=("cpu", "cuda", "tensorrt"),
        action="append",
        help="Execution providers for ONNX inference (repeat to set fallback order).",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=42,
        help="Seed used when generating blended inputs for the synthetic split.",
    )
    return parser.parse_args()


def iter_image_files(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        yield path


def load_image_np(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def load_image(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor


def compute_metrics(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    # torchmetrics expects tensors with a batch dimension.
    preds = a.unsqueeze(0)
    target = b.unsqueeze(0)
    psnr = peak_signal_noise_ratio(preds, target, data_range=1.0)
    ssim = structural_similarity_index_measure(preds, target, data_range=1.0)
    return psnr.item(), ssim.item()


def st_norm_cdf(x: np.ndarray) -> np.ndarray:
    return norm.cdf(x)


def gaussian_kernel(kernlen: int = 100, nsig: float = 1.0) -> np.ndarray:
    interval = (2 * nsig + 1.0) / kernlen
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernlen + 1)
    kern1d = np.diff(st_norm_cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel / kernel.max()


def create_vignetting_mask(size: int = 560, nsig: float = 3.0) -> np.ndarray:
    kernel = gaussian_kernel(size, nsig)
    return np.dstack((kernel, kernel, kernel))


def ensure_gaussian_mask(min_size: int) -> np.ndarray:
    global GAUSSIAN_MASK
    if GAUSSIAN_MASK is None or GAUSSIAN_MASK.shape[0] < min_size or GAUSSIAN_MASK.shape[1] < min_size:
        GAUSSIAN_MASK = create_vignetting_mask(max(min_size + 11, 600))
    return GAUSSIAN_MASK


def syn_data_with_rng(t: np.ndarray, r: np.ndarray, sigma: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    global GAUSSIAN_MASK
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(r, 0.0, 1.0)

    t_gamma = np.power(t, 2.2)
    r_gamma = np.power(r, 2.2)

    sz = int(2 * np.ceil(2 * sigma) + 1)
    if sz % 2 == 0:
        sz += 1
    r_blur = cv2.GaussianBlur(r_gamma, (sz, sz), sigma, sigma, 0)
    blend = r_blur + t_gamma

    att = 1.08 + float(rng.random()) / 10.0
    for i in range(3):
        maski = blend[:, :, i] > 1
        denominator = maski.sum() + 1e-6
        mean_i = max(1.0, float(np.sum(blend[:, :, i] * maski) / denominator))
        r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
    r_blur = np.clip(r_blur, 0, 1)

    h, w = r_blur.shape[:2]
    mask = ensure_gaussian_mask(max(h, w))
    if mask.shape[0] < h or mask.shape[1] < w:
        resize_w = max(mask.shape[1], w + 11)
        resize_h = max(mask.shape[0], h + 11)
        mask = cv2.resize(mask, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        GAUSSIAN_MASK = mask
        mask = GAUSSIAN_MASK

    neww = int(rng.integers(0, mask.shape[1] - w - 10))
    newh = int(rng.integers(0, mask.shape[0] - h - 10))
    alpha1 = mask[newh:newh + h, neww:neww + w, :]
    alpha2 = 1 - float(rng.random()) / 5.0
    r_blur_mask = r_blur * alpha1
    blend = r_blur_mask + t_gamma * alpha2

    t_out = np.power(t_gamma, 1 / 2.2)
    r_out = np.power(r_blur_mask, 1 / 2.2)
    blend_out = np.power(np.clip(blend, 0, 1), 1 / 2.2)
    blend_out = np.clip(blend_out, 0, 1)

    return t_out, r_out, blend_out


def resize_with_aspect(image: np.ndarray, width: int) -> np.ndarray:
    new_h = max(1, int(round((width / image.shape[1]) * image.shape[0])))
    return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_CUBIC)


def generate_synthetic_blend(
    transmission_path: Path,
    reflection_path: Path,
    rng: np.random.Generator,
    max_attempts: int = 5,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        transmission = load_image_np(transmission_path)
        reflection = load_image_np(reflection_path)
    except ValueError as exc:
        print(f"[WARN] {exc}")
        return None

    width_choices = list(range(SYNTHETIC_MIN_WIDTH, SYNTHETIC_MAX_WIDTH + 1, SYNTHETIC_WIDTH_STEP))

    for _ in range(max_attempts):
        width = int(rng.choice(width_choices))
        transmission_resized = resize_with_aspect(transmission, width)
        reflection_resized = resize_with_aspect(reflection, width)
        sigma = float(rng.choice(SYNTHETIC_SIGMAS))
        t_gt, _, blended = syn_data_with_rng(transmission_resized, reflection_resized, sigma, rng)
        return blended, t_gt
    print(f"[WARN] Unable to generate synthetic blend for {transmission_path.name} after {max_attempts} attempts.")
    return None


def _normalise_backbone(name: str) -> str:
    candidate = name.lower()
    if candidate not in BACKBONE_CHOICES:
        raise ValueError(
            f"Unsupported backbone '{name}'. Expected one of {BACKBONE_CHOICES}."
        )
    return candidate


def _resolve_backbone(meta: Optional[str], override: Optional[str]) -> str:
    if override:
        override_norm = _normalise_backbone(override)
        if meta:
            meta_norm = _normalise_backbone(meta)
            if meta_norm != override_norm:
                raise ValueError(
                    f"Checkpoint backbone '{meta_norm}' does not match requested '{override_norm}'."
                )
        return override_norm
    if meta:
        return _normalise_backbone(meta)
    raise ValueError(
        "Backbone is required for checkpoints without metadata. "
        "Specify --backbone explicitly."
    )


def infer_variant_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    for key in state_dict.keys():
        if key.startswith("rir_inner_scales") or key.startswith("rir_group_scales"):
            return "residual_in_residual_skips"
        if key.startswith("residual_scales") or key.startswith("output_skip_scale"):
            return "residual_skips"
    return "baseline"


def resolve_checkpoint_variant(
    metadata_variant: Optional[str],
    state_dict: Dict[str, torch.Tensor],
) -> str:
    if metadata_variant:
        return metadata_variant
    return infer_variant_from_state_dict(state_dict)


def state_dict_has_output_skip(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("output_skip_scale") for key in state_dict.keys())


def state_dict_uses_distributed_hypercolumns(state_dict: Dict[str, torch.Tensor]) -> bool:
    prefixes = (
        "feature_extractor._distributed_reduction_layers",
        "feature_extractor._distributed_post",
    )
    return any(key.startswith(prefixes) for key in state_dict.keys())


def infer_hypercolumn_reduction_scale(
    state_dict: Dict[str, torch.Tensor],
    default: int = 4,
) -> int:
    prefix = "feature_extractor._distributed_reduction_layers."
    suffix = ".weight"
    channel_pairs: List[Tuple[int, int]] = []
    for key, value in state_dict.items():
        if not (key.startswith(prefix) and key.endswith(suffix)):
            continue
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            continue
        out_channels = int(value.shape[0])
        in_channels = int(value.shape[1])
        if out_channels <= 0:
            continue
        channel_pairs.append((in_channels, out_channels))

    if not channel_pairs:
        return default

    lower_bound = max(math.ceil(in_ch / out_ch) for in_ch, out_ch in channel_pairs)
    upper_candidates = [
        (in_ch - 1) // (out_ch - 1)
        for in_ch, out_ch in channel_pairs
        if out_ch > 1
    ]

    if upper_candidates:
        upper_bound = min(upper_candidates)
        if upper_bound < lower_bound:
            warnings.warn(
                (
                    "Inconsistent distributed hypercolumn reductions detected; "
                    f"falling back to minimum feasible scale {lower_bound}."
                ),
                RuntimeWarning,
            )
            candidate_range = [lower_bound]
        else:
            candidate_range = range(upper_bound, lower_bound - 1, -1)
    else:
        candidate_range = [lower_bound]

    for scale in candidate_range:
        if scale < 1:
            continue
        if all(math.ceil(in_ch / scale) == out_ch for in_ch, out_ch in channel_pairs):
            return scale

    warnings.warn(
        "Unable to infer hypercolumn reduction scale from checkpoint; using default value.",
        RuntimeWarning,
    )
    return default


def build_generator_metadata(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    distributed = state_dict_uses_distributed_hypercolumns(state_dict)
    scale = infer_hypercolumn_reduction_scale(state_dict)
    return {
        "distributed_hypercolumn": distributed,
        "hypercolumn_reduction_scale": scale,
    }


def extract_state_dict_variant_and_backbone(
    state: Dict[str, Any],
    backbone_hint: Optional[str],
) -> Tuple[Dict[str, torch.Tensor], str, str]:
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        backbone = _resolve_backbone(state.get("backbone"), backbone_hint)
        state_dict = state["state_dict"]
        variant = resolve_checkpoint_variant(state.get("generator_variant"), state_dict)
        return state_dict, variant, backbone
    if "generator" in state and isinstance(state["generator"], dict):
        backbone = _resolve_backbone(state.get("backbone"), backbone_hint)
        state_dict = state["generator"]
        variant = resolve_checkpoint_variant(state.get("generator_variant"), state_dict)
        return state_dict, variant, backbone

    meta_backbone = state.get("backbone") if isinstance(state, dict) else None
    backbone = _resolve_backbone(meta_backbone, backbone_hint)
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint structure; expected a dict.")
    variant = infer_variant_from_state_dict(state)
    state_dict = state
    return state_dict, variant, backbone


def _resolve_checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in ("generator.pt", "checkpoint_latest.pt"):
            candidate_path = path / candidate
            if candidate_path.exists():
                return candidate_path
    raise FileNotFoundError(
        f"Checkpoint '{path}' not found. "
        "Provide a .pt file or a directory containing generator.pt/checkpoint_latest.pt."
    )


def load_generator_state_dict(
    checkpoint_path: Path,
    backbone_hint: Optional[str],
) -> Tuple[Dict[str, torch.Tensor], str, Dict[str, Any], str]:
    ckpt_file = _resolve_checkpoint_file(checkpoint_path)
    state = torch.load(ckpt_file, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format stored in {ckpt_file}.")
    state_dict, variant, backbone = extract_state_dict_variant_and_backbone(state, backbone_hint)
    metadata = build_generator_metadata(state_dict)
    return state_dict, variant, metadata, backbone


def create_feature_extractor(
    backbone: str,
    metadata: Dict[str, Any],
    ckpt_dir: Path,
) -> FeatureExtractorBase:
    distributed = bool(metadata.get("distributed_hypercolumn", False))
    reduction_scale = int(metadata.get("hypercolumn_reduction_scale", 4))
    ckpt_dir = ckpt_dir.expanduser()
    if backbone == "vgg19":
        return VGGFeatureExtractor(
            use_hyper=True,
            distributed_hypercolumn=distributed,
            hypercolumn_reduction_scale=reduction_scale,
        )
    if backbone == "hgnetv2":
        return HGNetFeatureExtractor(
            use_hyper=True,
            ckpt_root=ckpt_dir,
            distributed_hypercolumn=distributed,
            hypercolumn_reduction_scale=reduction_scale,
        )
    if backbone in DINOFeatureExtractor.CKPT_FILENAMES:
        return DINOFeatureExtractor(
            backbone,
            use_hyper=True,
            ckpt_root=ckpt_dir,
            distributed_hypercolumn=distributed,
            hypercolumn_reduction_scale=reduction_scale,
        )
    raise ValueError(f"Unsupported backbone '{backbone}'.")


def build_generator_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    variant: str,
    metadata: Dict[str, Any],
    backbone: str,
    ckpt_dir: Path,
    device: torch.device,
) -> HypercolumnGenerator:
    feature_extractor = create_feature_extractor(backbone, metadata, ckpt_dir)
    output_skip_scale = 0.0 if state_dict_has_output_skip(state_dict) else None
    if variant == "residual_in_residual_skips":
        generator: HypercolumnGenerator = ResidualInResidualHypercolumnGenerator(
            feature_extractor,
            residual_init=0.1,
            output_skip_init=output_skip_scale,
        )
    elif variant == "residual_skips":
        generator = ResidualHypercolumnGenerator(
            feature_extractor,
            residual_init=0.1,
            output_skip_init=output_skip_scale,
        )
    elif variant == "baseline":
        generator = HypercolumnGenerator(feature_extractor)
    else:
        raise ValueError(f"Unknown generator variant '{variant}'.")

    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()
    generator.feature_extractor.eval()
    return generator


def select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_numpy_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().float().clamp(0.0, 1.0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return arr


def numpy_to_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    return tensor.float()


class ModelRunner:
    def predict(self, image: torch.Tensor, image_path: Path) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class TorchGeneratorRunner(ModelRunner):
    def __init__(
        self,
        checkpoint_path: Path,
        backbone: Optional[str],
        ckpt_dir: Path,
        device_arg: Optional[str],
    ):
        self.device = select_device(device_arg)
        state_dict, variant, metadata, resolved_backbone = load_generator_state_dict(
            checkpoint_path,
            backbone_hint=backbone,
        )
        self.generator = build_generator_from_state_dict(
            state_dict,
            variant,
            metadata,
            resolved_backbone,
            ckpt_dir,
            self.device,
        )

    def predict(self, image: torch.Tensor, image_path: Path) -> torch.Tensor:
        tensor = image.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            transmission, _ = self.generator(tensor)
        return transmission.squeeze(0).detach().cpu().clamp(0.0, 1.0)


def infer_input_size(session: ort.InferenceSession) -> Optional[Tuple[int, int]]:
    shape = session.get_inputs()[0].shape
    if len(shape) != 4:
        return None
    height, width = shape[2], shape[3]
    try:
        return int(height), int(width)
    except (TypeError, ValueError):
        return None


def resize_for_model(image: np.ndarray, size_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    if size_hw is None:
        return image
    height, width = size_hw
    if image.shape[0] == height and image.shape[1] == width:
        return image
    interpolation = (
        cv2.INTER_AREA if image.shape[0] > height or image.shape[1] > width else cv2.INTER_CUBIC
    )
    return cv2.resize(image, (width, height), interpolation=interpolation)


def prepare_input_tensor_np(image_rgb: np.ndarray) -> np.ndarray:
    tensor = image_rgb.astype(np.float32)
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.expand_dims(tensor, axis=0)


def postprocess_output_np(
    output: np.ndarray,
    original_size: Tuple[int, int],
) -> np.ndarray:
    data = np.squeeze(output, axis=0)
    data = np.transpose(data, (1, 2, 0))
    if (data.shape[1], data.shape[0]) != original_size:
        data = cv2.resize(data, original_size, interpolation=cv2.INTER_CUBIC)
    return np.clip(data, 0.0, 1.0)


def resolve_providers(requested: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not requested:
        return None
    provider_map = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    available = set(ort.get_available_providers())
    providers: List[str] = []
    for name in requested:
        key = name.lower()
        if key not in provider_map:
            raise ValueError(f"Unsupported provider '{name}'. Choose from cpu/cuda/tensorrt.")
        resolved = provider_map[key]
        if resolved not in available:
            raise ValueError(
                f"Execution provider '{resolved}' is not available. "
                f"Available providers: {sorted(available)}"
            )
        if resolved not in providers:
            providers.append(resolved)
    if (
        "CPUExecutionProvider" in available
        and providers
        and providers[-1] != "CPUExecutionProvider"
    ):
        providers.append("CPUExecutionProvider")
    return providers


class OnnxGeneratorRunner(ModelRunner):
    def __init__(self, model_path: Path, providers: Optional[Sequence[str]]):
        providers_list = resolve_providers(providers)
        self.model_path = model_path.expanduser()
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers_list or None,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.model_hw = infer_input_size(self.session)

    def predict(self, image: torch.Tensor, image_path: Path) -> torch.Tensor:
        rgb = tensor_to_numpy_rgb(image)
        original_size = (rgb.shape[1], rgb.shape[0])
        resized = resize_for_model(rgb, self.model_hw)
        tensor = prepare_input_tensor_np(resized)
        raw_outputs = self.session.run(self.output_names, {self.input_name: tensor})
        outputs = dict(zip(self.output_names, raw_outputs))
        transmission = outputs.get("transmission")
        if transmission is None:
            transmission = raw_outputs[0]
        processed = postprocess_output_np(transmission, original_size)
        return numpy_to_tensor(processed).clamp(0.0, 1.0)


def evaluate_synthetic_split(
    dataset_root: Path,
    skip_missing: bool,
    model_runner: Optional[ModelRunner],
    max_samples: Optional[int],
    synthetic_seed: int,
) -> List[Tuple[str, float, float]]:
    if model_runner is None:
        raise ValueError(
            "Synthetic split evaluation requires a generator (--checkpoint or --onnx-model)."
        )
    trans_dir = dataset_root / "synthetic" / "transmission_layer"
    refl_dir = dataset_root / "synthetic" / "reflection_layer"
    if not trans_dir.exists():
        raise FileNotFoundError(f"Missing synthetic transmission directory: {trans_dir}")
    if not refl_dir.exists():
        raise FileNotFoundError(f"Missing synthetic reflection directory: {refl_dir}")

    files = [path for path in sorted(trans_dir.iterdir()) if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS]
    if max_samples is not None and max_samples > 0:
        files = files[:max_samples]

    per_image: List[Tuple[str, float, float]] = []
    for idx, trans_path in enumerate(
        tqdm(files, desc="synthetic: blended vs transmission_layer", total=len(files) if files else None)
    ):
        refl_path = refl_dir / trans_path.name
        if not refl_path.exists():
            message = f"No matching reflection file for {trans_path.name} in {refl_dir}"
            if skip_missing:
                print(f"[WARN] {message}")
                continue
            raise FileNotFoundError(message)

        rng = np.random.default_rng(synthetic_seed + idx)
        sample = generate_synthetic_blend(trans_path, refl_path, rng)
        if sample is None:
            continue
        blended_np, target_np = sample
        blended_tensor = numpy_to_tensor(blended_np)
        target_tensor = numpy_to_tensor(target_np)
        try:
            pred_tensor = model_runner.predict(blended_tensor, trans_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to run inference on '{trans_path.name}': {exc}")
            continue
        if pred_tensor.shape[-2:] != target_tensor.shape[-2:]:
            pred_tensor = F.interpolate(
                pred_tensor.unsqueeze(0),
                size=target_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        psnr, ssim = compute_metrics(pred_tensor.clamp(0.0, 1.0), target_tensor)
        per_image.append((trans_path.name, psnr, ssim))
    return per_image


def evaluate_split(
    dataset_root: Path,
    split: str,
    skip_missing: bool = False,
    model_runner: Optional[ModelRunner] = None,
    max_samples: Optional[int] = None,
    synthetic_seed: int = 42,
) -> List[Tuple[str, float, float]]:
    if split == "synthetic":
        return evaluate_synthetic_split(
            dataset_root,
            skip_missing=skip_missing,
            model_runner=model_runner,
            max_samples=max_samples,
            synthetic_seed=synthetic_seed,
        )
    src_dir_name, tgt_dir_name = PAIR_CONFIG[split]
    src_dir = dataset_root / split / src_dir_name
    tgt_dir = dataset_root / split / tgt_dir_name

    if not src_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {src_dir}")
    if not tgt_dir.exists():
        raise FileNotFoundError(f"Missing target directory: {tgt_dir}")

    per_image: List[Tuple[str, float, float]] = []

    src_files = list(iter_image_files(src_dir))
    if max_samples is not None and max_samples > 0:
        src_files = src_files[:max_samples]

    for src_path in tqdm(
        src_files,
        desc=f"{split}: {src_dir_name} vs {tgt_dir_name}",
        total=len(src_files) if src_files else None,
    ):
        tgt_path = tgt_dir / src_path.name
        if not tgt_path.exists():
            message = f"No matching file for {src_path.name} in {tgt_dir}"
            if skip_missing:
                print(f"[WARN] {message}")
                continue
            raise FileNotFoundError(message)

        try:
            src_tensor = load_image(src_path)
            tgt_tensor = load_image(tgt_path)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue

        if model_runner is not None:
            try:
                pred_tensor = model_runner.predict(src_tensor, src_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to run inference on '{src_path.name}': {exc}")
                continue
        else:
            pred_tensor = src_tensor

        if pred_tensor.shape[-2:] != tgt_tensor.shape[-2:]:
            pred_tensor = F.interpolate(
                pred_tensor.unsqueeze(0),
                size=tgt_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        psnr, ssim = compute_metrics(pred_tensor.clamp(0.0, 1.0), tgt_tensor)
        per_image.append((src_path.name, psnr, ssim))

    return per_image


def print_summary(
    split: str,
    src_dir: str,
    tgt_dir: str,
    per_image_metrics: Sequence[Tuple[str, float, float]],
) -> None:
    if not per_image_metrics:
        print(f"[INFO] No comparable samples found for {split}.")
        return

    psnr_values = [m[1] for m in per_image_metrics]
    ssim_values = [m[2] for m in per_image_metrics]

    def format_stats(values: Sequence[float]) -> str:
        return (
            f"mean={sum(values) / len(values):.4f}, "
            f"min={min(values):.4f}, max={max(values):.4f}"
        )

    print(
        f"[RESULT] split={split} ({src_dir} vs {tgt_dir}) "
        f"samples={len(per_image_metrics)}"
    )
    print(f"         PSNR  -> {format_stats(psnr_values)}")
    print(f"         SSIM  -> {format_stats(ssim_values)}")


def save_csv(
    path: Path,
    rows: Sequence[Tuple[str, str, str, str, float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["split", "source_dir", "target_dir", "filename", "psnr", "ssim"])
        writer.writerows(rows)
    print(f"[INFO] Wrote per-image metrics to {path}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if args.max_samples is not None and args.max_samples <= 0:
        print("[WARN] --max-samples must be > 0; ignoring the limit.")
        args.max_samples = None

    splits_to_run = (
        list(PAIR_CONFIG.keys()) if args.subset == "all" else [args.subset]
    )

    model_runner: Optional[ModelRunner] = None
    if args.checkpoint:
        checkpoint_path = args.checkpoint.expanduser()
        print(f"[INFO] Loading PyTorch checkpoint from {checkpoint_path}")
        model_runner = TorchGeneratorRunner(
            checkpoint_path,
            backbone=args.backbone,
            ckpt_dir=args.ckpt_dir,
            device_arg=args.device,
        )
    elif args.onnx_model:
        model_path = args.onnx_model.expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model '{model_path}' not found.")
        print(f"[INFO] Loading ONNX model from {model_path}")
        model_runner = OnnxGeneratorRunner(model_path, args.providers)

    csv_rows: List[Tuple[str, str, str, str, float, float]] = []

    for split in splits_to_run:
        if split == "synthetic":
            src_dir_name = "synthetic_blended"
            tgt_dir_name = PAIR_CONFIG["synthetic"][1]
        else:
            src_dir_name, tgt_dir_name = PAIR_CONFIG[split]
        metrics = evaluate_split(
            dataset_root,
            split,
            skip_missing=args.skip_missing,
            model_runner=model_runner,
            max_samples=args.max_samples,
            synthetic_seed=args.synthetic_seed,
        )
        print_summary(split, src_dir_name, tgt_dir_name, metrics)

        for filename, psnr, ssim in metrics:
            csv_rows.append((split, src_dir_name, tgt_dir_name, filename, psnr, ssim))

    if args.output_csv and csv_rows:
        save_csv(args.output_csv, csv_rows)


if __name__ == "__main__":
    main()

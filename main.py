import argparse
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import amp
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from discriminator import PatchDiscriminator
from losses import (compute_exclusion_loss, compute_l1_loss,
                    compute_perceptual_loss)
from models import (DINOFeatureExtractor, FeatureExtractorBase,
                    FeatureProjector, HypercolumnGenerator,
                    ResidualHypercolumnGenerator, VGGFeatureExtractor,
                    HGNetFeatureExtractor)


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp'}
GAUSSIAN_MASK = None
BACKBONE_CHOICES = ["vgg19", "hgnetv2", "dinov3_vitt", "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16"]
VISUAL_SNAPSHOTS_PER_EPOCH = 0
GRAD_CLIP_NORM = 5.0
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "runs"

GENERATOR_VARIANT_BASELINE = "baseline"
GENERATOR_VARIANT_RESIDUAL = "residual_skips"


def generator_variant_name(use_residual: bool) -> str:
    return GENERATOR_VARIANT_RESIDUAL if use_residual else GENERATOR_VARIANT_BASELINE


def generator_variant_from_module(generator: HypercolumnGenerator) -> str:
    if isinstance(generator, ResidualHypercolumnGenerator):
        return GENERATOR_VARIANT_RESIDUAL
    return GENERATOR_VARIANT_BASELINE


def infer_variant_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    for key in state_dict.keys():
        if key.startswith("residual_scales") or key.startswith("output_skip_scale"):
            return GENERATOR_VARIANT_RESIDUAL
    return GENERATOR_VARIANT_BASELINE


def resolve_checkpoint_variant(
    metadata_variant: Optional[str],
    state_dict: Dict[str, torch.Tensor],
) -> str:
    if metadata_variant:
        return metadata_variant
    return infer_variant_from_state_dict(state_dict)


def build_generator(
    feature_extractor: FeatureExtractorBase,
    residual_skips: bool,
    residual_init: float,
    output_skip_scale: Optional[float],
) -> HypercolumnGenerator:
    if residual_skips:
        return ResidualHypercolumnGenerator(
            feature_extractor,
            residual_init=residual_init,
            output_skip_init=output_skip_scale,
        )
    if output_skip_scale is not None:
        warnings.warn(
            "--output_skip_scale is only used when --residual_skips is enabled; ignoring the value.",
            RuntimeWarning,
        )
    return HypercolumnGenerator(feature_extractor)


def state_dict_has_output_skip(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("output_skip_scale") for key in state_dict.keys())


def _validate_checkpoint_backbone(meta: Optional[str], backbone: str) -> None:
    if meta is None:
        return
    meta_norm = meta.lower()
    backbone_norm = backbone.lower()
    if meta_norm != backbone_norm:
        raise ValueError(
            f"Checkpoint backbone ({meta_norm}) does not match requested backbone ({backbone_norm})."
        )


def extract_state_dict_and_variant(
    state: Dict[str, Any],
    backbone: str,
) -> Tuple[Dict[str, torch.Tensor], str]:
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        _validate_checkpoint_backbone(state.get("backbone"), backbone)
        state_dict = state["state_dict"]
        variant = resolve_checkpoint_variant(state.get("generator_variant"), state_dict)
        return state_dict, variant
    if "generator" in state and isinstance(state["generator"], dict):
        _validate_checkpoint_backbone(state.get("backbone"), backbone)
        state_dict = state["generator"]
        variant = resolve_checkpoint_variant(state.get("generator_variant"), state_dict)
        return state_dict, variant
    state_dict = state
    variant = infer_variant_from_state_dict(state_dict)
    return state_dict, variant


def load_generator_state_dict_from_artifact(
    artifact_path: Path,
    device: torch.device,
    backbone: str,
) -> Tuple[Dict[str, torch.Tensor], str]:
    path = artifact_path
    if path.is_dir():
        generator_file = path / "generator.pt"
        if generator_file.exists():
            state = torch.load(generator_file, map_location=device)
            if not isinstance(state, dict):
                raise ValueError(f"Unsupported generator checkpoint format in {generator_file}.")
            return extract_state_dict_and_variant(state, backbone)
        checkpoint_file = path / "checkpoint_latest.pt"
        if checkpoint_file.exists():
            state = torch.load(checkpoint_file, map_location=device)
            if not isinstance(state, dict):
                raise ValueError(f"Unsupported checkpoint format in {checkpoint_file}.")
            return extract_state_dict_and_variant(state, backbone)
        raise FileNotFoundError(
            f"No generator checkpoint found under {path}. Expected 'generator.pt' or 'checkpoint_latest.pt'."
        )
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported generator checkpoint format in {path}.")
    return extract_state_dict_and_variant(state, backbone)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perceptual reflection removal (PyTorch)")
    parser.add_argument("--exp_name", default="default", help="experiment name (saved under runs/)")
    parser.add_argument("--data_syn_dir", default="", help="comma separated list of synthetic dataset roots")
    parser.add_argument("--data_real_dir", default="", help="comma separated list of real dataset roots")
    parser.add_argument("--save_model_freq", type=int, default=1, help="save model frequency (epochs)")
    parser.add_argument("--keep_checkpoint_history", type=int, default=20, help="number of saved checkpoint epochs to retain (0 keeps all)")
    parser.add_argument("--test_only", action="store_true", help="run inference only (skip training)")
    parser.add_argument("--resume", action="store_true", help="resume training from the checkpoint stored under runs/--exp_name")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--device", default=None, help="device identifier (e.g. cuda:0)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--log_interval", type=int, default=100, help="iterations between logging updates")
    parser.add_argument("--test_dir", default="./test_images", help="comma separated list of directories for evaluation")
    parser.add_argument("--backbone", default="dinov3_vits16", choices=BACKBONE_CHOICES, help="feature backbone for hypercolumns and perceptual loss")
    parser.add_argument("--ckpt_dir", default="ckpts", help="directory for optional pretrained backbone weights")
    parser.add_argument("--ckpt_file", default="", help="path to generator checkpoint used for weight initialization")
    parser.add_argument("--use_amp", action="store_true", help="enable automatic mixed precision during train/test")
    parser.add_argument("--use_distributed_hypercolumn", action="store_true", help="enable distributed hypercolumn compression (per-layer reduction and 64-channel fusion)")
    parser.add_argument("--hypercolumn_channel_reduction_scale", type=int, default=4, help="Divisor used for distributed hypercolumn channel reduction (must be >= 1).")
    parser.add_argument("--distill_teacher_backbone", default=None, choices=BACKBONE_CHOICES, help="backbone for the frozen teacher generator used in distillation")
    parser.add_argument("--distill_teacher_checkpoint", default="", help="path to a teacher generator checkpoint (.pt or checkpoint directory)")
    parser.add_argument("--distill_feature_weight", type=float, default=0.05, help="weight for feature-map distillation loss (MSE)")
    parser.add_argument("--distill_pixel_weight", type=float, default=0.02, help="weight for teacher output distillation loss (L1)")
    parser.add_argument("--residual_skips", action="store_true", help="enable residual skip connections after the hypercolumn stem")
    parser.add_argument("--residual_init", type=float, default=0.1, help="initial residual scale when --residual_skips is enabled")
    parser.add_argument("--output_skip_scale", type=float, default=None, help="initial transmission skip scale; requires --residual_skips")
    args = parser.parse_args()
    if args.hypercolumn_channel_reduction_scale < 1:
        parser.error("--hypercolumn_channel_reduction_scale must be >= 1")
    return args


def get_experiment_dir(exp_name: str) -> Path:
    exp_path = Path(exp_name)
    if exp_path.is_absolute():
        return exp_path
    if exp_path.parts and exp_path.parts[0] == "runs":
        exp_path = Path(*exp_path.parts[1:])
    return RUNS_ROOT / exp_path


def get_results_dir(exp_name: str) -> Path:
    """Base directory for per-epoch validation outputs."""
    return get_experiment_dir(exp_name)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def create_feature_extractor(
    backbone: str,
    use_hyper: bool,
    ckpt_dir: Path,
    distributed_hypercolumn: bool = False,
    hypercolumn_reduction_scale: int = 4,
) -> FeatureExtractorBase:
    name = backbone.lower()
    if name == "vgg19":
        return VGGFeatureExtractor(
            use_hyper=use_hyper,
            distributed_hypercolumn=distributed_hypercolumn,
            hypercolumn_reduction_scale=hypercolumn_reduction_scale,
        )
    if name == "hgnetv2":
        return HGNetFeatureExtractor(
            use_hyper=use_hyper,
            ckpt_root=ckpt_dir,
            distributed_hypercolumn=distributed_hypercolumn,
            hypercolumn_reduction_scale=hypercolumn_reduction_scale,
        )
    if name in DINOFeatureExtractor.CKPT_FILENAMES:
        return DINOFeatureExtractor(
            name,
            use_hyper=use_hyper,
            ckpt_root=ckpt_dir,
            distributed_hypercolumn=distributed_hypercolumn,
            hypercolumn_reduction_scale=hypercolumn_reduction_scale,
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def collect_roots(path_string: str) -> List[Path]:
    roots = []
    for fragment in path_string.split(','):
        fragment = fragment.strip()
        if fragment:
            roots.append(resolve_path(fragment))
    return roots


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def prepare_synthetic_data(roots: Iterable[Path]) -> Tuple[List[Path], List[Path]]:
    transmissions: List[Path] = []
    reflections: List[Path] = []
    for root in roots:
        t_root = root / "transmission_layer"
        r_root = root / "reflection_layer"
        if not t_root.exists() or not r_root.exists():
            continue
        for file in sorted(t_root.rglob("*")):
            if file.is_file() and is_image_file(file):
                reflection_path = r_root / file.name
                if reflection_path.exists():
                    transmissions.append(file)
                    reflections.append(reflection_path)
    return transmissions, reflections


def prepare_real_data(roots: Iterable[Path]) -> Tuple[List[Path], List[Path]]:
    blended: List[Path] = []
    transmissions: List[Path] = []
    for root in roots:
        t_root = root / "transmission_layer"
        b_root = root / "blended"
        if not t_root.exists() or not b_root.exists():
            continue
        for file in sorted(t_root.rglob("*")):
            if file.is_file() and is_image_file(file):
                blended_path = b_root / file.name
                if blended_path.exists():
                    transmissions.append(file)
                    blended.append(blended_path)
    return blended, transmissions


def prepare_test_images(roots: Iterable[Path]) -> List[Path]:
    images: List[Path] = []
    for root in roots:
        if root.is_file() and is_image_file(root):
            images.append(root)
        elif root.is_dir():
            for file in sorted(root.rglob("*")):
                if file.is_file() and is_image_file(file):
                    images.append(file)
    return images


def gaussian_kernel(kernlen: int = 100, nsig: float = 1.0) -> np.ndarray:
    interval = (2 * nsig + 1.0) / kernlen
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernlen + 1)
    kern1d = np.diff(st_norm_cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel / kernel.max()


def st_norm_cdf(x: np.ndarray) -> np.ndarray:
    import scipy.stats as st
    return st.norm.cdf(x)


def create_vignetting_mask(size: int = 560, nsig: float = 3) -> np.ndarray:
    kernel = gaussian_kernel(size, nsig)
    return np.dstack((kernel, kernel, kernel))


def syn_data(t: np.ndarray, r: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    global GAUSSIAN_MASK
    if GAUSSIAN_MASK is None or GAUSSIAN_MASK.shape[0] < max(t.shape[0], t.shape[1], r.shape[0], r.shape[1]):
        GAUSSIAN_MASK = create_vignetting_mask()

    t = np.clip(t, 0.0, 1.0)
    r = np.clip(r, 0.0, 1.0)

    t_gamma = np.power(t, 2.2)
    r_gamma = np.power(r, 2.2)

    sz = int(2 * np.ceil(2 * sigma) + 1)
    if sz % 2 == 0:
        sz += 1
    r_blur = cv2.GaussianBlur(r_gamma, (sz, sz), sigma, sigma, 0)
    blend = r_blur + t_gamma

    att = 1.08 + np.random.random() / 10.0

    for i in range(3):
        maski = blend[:, :, i] > 1
        denominator = maski.sum() + 1e-6
        mean_i = max(1.0, np.sum(blend[:, :, i] * maski) / denominator)
        r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
    r_blur = np.clip(r_blur, 0, 1)

    h, w = r_blur.shape[:2]
    mask = GAUSSIAN_MASK
    if mask.shape[0] < h or mask.shape[1] < w:
        resize_w = max(mask.shape[1], w + 11)
        resize_h = max(mask.shape[0], h + 11)
        mask = cv2.resize(mask, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        GAUSSIAN_MASK = mask
    neww = np.random.randint(0, mask.shape[1] - w - 10)
    newh = np.random.randint(0, mask.shape[0] - h - 10)
    alpha1 = mask[newh:newh + h, neww:neww + w, :]
    alpha2 = 1 - np.random.random() / 5.0
    r_blur_mask = r_blur * alpha1
    blend = r_blur_mask + t_gamma * alpha2

    t_out = np.power(t_gamma, 1 / 2.2)
    r_out = np.power(r_blur_mask, 1 / 2.2)
    blend_out = np.power(np.clip(blend, 0, 1), 1 / 2.2)
    blend_out = np.clip(blend_out, 0, 1)

    return t_out, r_out, blend_out


def load_image(path: Path) -> Optional[np.ndarray]:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return np.float32(img_rgb) / 255.0


def resize_with_aspect(img: np.ndarray, width: int) -> np.ndarray:
    new_w = width
    new_h = max(1, int(round((new_w / img.shape[1]) * img.shape[0])))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float()


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().float()
    tensor = torch.clamp(tensor, 0.0, 1.0)
    arr = tensor.permute(1, 2, 0).numpy()
    return (arr * 255.0).astype(np.uint8)


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0 + 1e-5:
            scaled = np.clip(image, 0.0, 1.0) * 255.0
        else:
            scaled = np.clip(image, 0.0, 255.0)
    else:
        scaled = np.clip(image, 0, 255)
    return scaled.astype(np.uint8)


def write_rgb_image(path: Path, image: np.ndarray) -> None:
    data = to_uint8(image)
    if data.ndim == 3 and data.shape[2] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), data)


def prepare_synthetic_sample(idx: int, transmissions: List[Path], reflections: List[Path], sigmas: List[float]) -> Optional[Dict]:
    syn_t = load_image(transmissions[idx])
    syn_r = load_image(reflections[idx])
    if syn_t is None or syn_r is None:
        return None

    min_width, max_width, width_step = 256, 640, 16
    max_step = (max_width - min_width) // width_step
    new_w = min_width + width_step * random.randint(0, max_step)
    syn_t = resize_with_aspect(syn_t, new_w)
    syn_r = resize_with_aspect(syn_r, new_w)

    sigma = random.choice(sigmas)
    if syn_t.mean() * 0.5 > syn_r.mean():
        return None

    t_gt, r_gt, blended = syn_data(syn_t, syn_r, sigma)
    sample = {
        "input": blended,
        "target_t": t_gt,
        "target_r": r_gt,
        "is_syn": True,
        "file_id": transmissions[idx].stem,
    }
    return sample


def prepare_real_sample(idx: int, blends: List[Path], transmissions: List[Path]) -> Optional[Dict]:
    input_img = load_image(blends[idx])
    target_t = load_image(transmissions[idx])
    if input_img is None or target_t is None:
        return None

    min_width, max_width, width_step = 256, 640, 16
    max_step = (max_width - min_width) // width_step
    new_w = min_width + width_step * random.randint(0, max_step)
    input_img = resize_with_aspect(input_img, new_w)
    target_t = resize_with_aspect(target_t, new_w)

    sample = {
        "input": input_img,
        "target_t": target_t,
        "target_r": target_t,
        "is_syn": False,
        "file_id": blends[idx].stem,
    }
    return sample


def ensure_valid_sample(sample: Dict) -> bool:
    if sample["target_r"].max() < 0.15 or sample["target_t"].max() < 0.15:
        return False
    if sample["input"].max() < 0.1:
        return False
    return True


def save_images(epoch_dir: Path, file_id: str, input_img: np.ndarray,
                pred_t: np.ndarray, pred_r: np.ndarray,
                index: Optional[int] = None) -> None:
    prefix = f"{index:04d}_" if index is not None else ""
    result_dir = epoch_dir / f"{prefix}{file_id}"
    result_dir.mkdir(parents=True, exist_ok=True)

    write_rgb_image(result_dir / "input.png", input_img)
    write_rgb_image(result_dir / "t_output.png", pred_t)
    write_rgb_image(result_dir / "r_output.png", pred_r)


def save_checkpoint(exp_dir: Path, epoch: int, generator: HypercolumnGenerator,
                    discriminator: PatchDiscriminator, optimizer_g, optimizer_d,
                    backbone: str, feature_projector: Optional[nn.Module] = None) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    variant = generator_variant_from_module(generator)
    state = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "backbone": backbone,
        "generator_variant": variant,
    }
    if feature_projector is not None:
        state["feature_projector"] = feature_projector.state_dict()
    torch.save(state, exp_dir / "checkpoint_latest.pt")
    torch.save({
        "state_dict": generator.state_dict(),
        "backbone": backbone,
        "generator_variant": variant,
    }, exp_dir / "generator.pt")


def evaluate_samples(
    generator: HypercolumnGenerator,
    device: torch.device,
    images: List[Path],
    results_root: Path,
    epoch: int,
    use_amp: bool
) -> None:
    if not images:
        return

    was_training = generator.training
    generator.eval()
    generator.feature_extractor.eval()

    epoch_dir = results_root / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(images, start=1):
        img = load_image(image_path)
        if img is None:
            continue
        tensor = image_to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_t, pred_r = generator(tensor)
        pred_t_img = tensor_to_image(pred_t[0])
        pred_r_img = tensor_to_image(pred_r[0])

        subdir = epoch_dir / f"{idx:04d}_{image_path.stem}"
        subdir.mkdir(parents=True, exist_ok=True)
        write_rgb_image(subdir / "input.png", img)
        write_rgb_image(subdir / "t_output.png", pred_t_img)
        write_rgb_image(subdir / "r_output.png", pred_r_img)

    if was_training:
        generator.train()
        generator.feature_extractor.eval()


def prune_checkpoints(exp_dir: Path, max_keep: int) -> None:
    if max_keep <= 0:
        return
    def is_epoch_dir(path: Path) -> bool:
        name = path.name
        if name.isdigit():
            return True
        if name.startswith("epoch_") and name[6:].isdigit():
            return True
        return False

    checkpoint_dirs = sorted(
        [path for path in exp_dir.iterdir() if path.is_dir() and is_epoch_dir(path)]
    )
    excess = len(checkpoint_dirs) - max_keep
    for old_dir in checkpoint_dirs[:excess]:
        shutil.rmtree(old_dir, ignore_errors=True)


def resume_from_checkpoint(
    exp_dir: Path, device: torch.device,
    generator: HypercolumnGenerator,
    discriminator: PatchDiscriminator,
    optimizer_g, optimizer_d,
    backbone: str,
    expected_variant: str,
    feature_projector: Optional[nn.Module] = None
) -> int:
    ckpt_path = exp_dir / "checkpoint_latest.pt"
    if not ckpt_path.exists():
        return 1
    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_backbone = checkpoint.get("backbone", "vgg19")
    if ckpt_backbone != backbone:
        raise ValueError(
            f"Checkpoint backbone ({ckpt_backbone}) does not match requested backbone ({backbone})."
        )
    generator_state = checkpoint["generator"]
    ckpt_variant = resolve_checkpoint_variant(checkpoint.get("generator_variant"), generator_state)
    if ckpt_variant != expected_variant:
        raise ValueError(
            f"Checkpoint generator variant ({ckpt_variant}) does not match requested variant ({expected_variant})."
        )
    generator.load_state_dict(generator_state)
    discriminator.load_state_dict(checkpoint["discriminator"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    if feature_projector is not None and "feature_projector" in checkpoint:
        feature_projector.load_state_dict(checkpoint["feature_projector"])
    return int(checkpoint.get("epoch", 0)) + 1


def load_generator_for_inference(
    generator: HypercolumnGenerator,
    exp_dir: Path,
    device: torch.device,
    backbone: str,
    expected_variant: Optional[str] = None,
) -> None:
    state_dict, variant = load_generator_state_dict_from_artifact(exp_dir, device, backbone)
    if expected_variant and variant != expected_variant:
        raise ValueError(
            f"Generator checkpoint variant ({variant}) does not match requested variant ({expected_variant})."
        )
    generator.load_state_dict(state_dict)


def load_generator_weights_from_path(
    generator: HypercolumnGenerator,
    checkpoint_path: Path,
    device: torch.device,
    backbone: str,
    expected_variant: Optional[str] = None,
) -> None:
    state_dict, variant = load_generator_state_dict_from_artifact(checkpoint_path, device, backbone)
    if expected_variant and variant != expected_variant:
        raise ValueError(
            f"Teacher checkpoint generator variant ({variant}) does not match requested variant ({expected_variant})."
        )
    generator.load_state_dict(state_dict)



def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    use_hyper = True
    ckpt_root = resolve_path(args.ckpt_dir)
    exp_dir = get_experiment_dir(args.exp_name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = exp_dir / "train.log"
    log_file = open(log_path, "a", encoding="utf-8")
    writer = SummaryWriter(log_dir=str(exp_dir))

    def log(message: str) -> None:
        print(message)
        log_file.write(message + "\\n")
        log_file.flush()

    teacher_generator: Optional[HypercolumnGenerator] = None
    feature_projector: Optional[FeatureProjector] = None
    distill_feature_layers: List[str] = []
    feature_distill_weight = float(args.distill_feature_weight)
    pixel_distill_weight = float(args.distill_pixel_weight)

    try:
        feature_extractor = create_feature_extractor(
            args.backbone,
            use_hyper,
            ckpt_root,
            distributed_hypercolumn=args.use_distributed_hypercolumn,
            hypercolumn_reduction_scale=args.hypercolumn_channel_reduction_scale,
        )
        generator = build_generator(
            feature_extractor,
            residual_skips=args.residual_skips,
            residual_init=args.residual_init,
            output_skip_scale=args.output_skip_scale,
        ).to(device)
        discriminator = PatchDiscriminator().to(device)
        feature_extractor = generator.feature_extractor
        feature_extractor.eval()
        log(f"[i] Experiment directory: {exp_dir}")
        log(f"[i] Using backbone: {args.backbone}")
        log(
            f"[i] Hypercolumn channel reduction scale: {feature_extractor.hypercolumn_reduction_scale}"
        )
        variant_name = generator_variant_from_module(generator)
        log(f"[i] Generator variant: {variant_name}")
        output_skip_param = getattr(generator, "output_skip_scale", None)
        if output_skip_param is not None:
            log(f"[i] Initial output skip scale: {float(output_skip_param.item()):.4f}")
        if args.ckpt_file:
            if args.resume:
                log("[w] --ckpt_file is ignored because --resume is set; resuming from runs/ checkpoints.")
            else:
                pretrained_path = resolve_path(args.ckpt_file)
                state_dict, ckpt_variant = load_generator_state_dict_from_artifact(
                    pretrained_path,
                    device,
                    args.backbone,
                )
                if ckpt_variant != variant_name:
                    raise ValueError(
                        f"Pretrained checkpoint generator variant ({ckpt_variant}) does not match requested variant ({variant_name})."
                    )
                generator.load_state_dict(state_dict)
                log(f"[i] Loaded generator weights from {pretrained_path}")

        use_distillation = (feature_distill_weight > 0.0 or pixel_distill_weight > 0.0)
        if use_distillation and (not args.distill_teacher_backbone or not args.distill_teacher_checkpoint):
            log("[w] Distillation weights are non-zero, but teacher backbone/checkpoint not provided; disabling distillation.")
            feature_distill_weight = 0.0
            pixel_distill_weight = 0.0
            use_distillation = False
        if use_distillation:
            teacher_feature_extractor = create_feature_extractor(
                args.distill_teacher_backbone,
                use_hyper,
                ckpt_root,
            )
            teacher_ckpt_path = resolve_path(args.distill_teacher_checkpoint)
            teacher_state_dict, teacher_variant = load_generator_state_dict_from_artifact(
                teacher_ckpt_path,
                device,
                args.distill_teacher_backbone,
            )
            teacher_residual = teacher_variant == GENERATOR_VARIANT_RESIDUAL
            teacher_has_output_skip = state_dict_has_output_skip(teacher_state_dict)
            teacher_generator = build_generator(
                teacher_feature_extractor,
                residual_skips=teacher_residual,
                residual_init=0.0,
                output_skip_scale=0.0 if teacher_residual and teacher_has_output_skip else None,
            ).to(device)
            teacher_generator.load_state_dict(teacher_state_dict)
            teacher_generator.eval()
            for param in teacher_generator.parameters():
                param.requires_grad = False
            log(
                f"[i] Loaded teacher generator ({args.distill_teacher_backbone}, variant={teacher_variant}) "
                f"from {teacher_ckpt_path}"
            )
            if feature_distill_weight > 0.0:
                teacher_dims = teacher_generator.feature_extractor.layer_dims
                student_dims = generator.feature_extractor.layer_dims
                candidate_layers = list(generator.feature_extractor.hyper_layers)
                if "final" in teacher_dims and "final" in student_dims:
                    candidate_layers.append("final")
                distill_feature_layers = [
                    layer for layer in candidate_layers
                    if layer in teacher_dims and layer in student_dims
                ]
                if distill_feature_layers:
                    feature_projector = FeatureProjector(teacher_dims, student_dims, distill_feature_layers).to(device)
                    log(f"[i] Feature distillation layers: {', '.join(distill_feature_layers)}")
                else:
                    log("[w] No overlapping feature layers found; disabling feature distillation.")
                    feature_distill_weight = 0.0

        feature_distill_enabled = feature_projector is not None and feature_distill_weight > 0.0
        pixel_distill_enabled = teacher_generator is not None and pixel_distill_weight > 0.0

        use_amp = bool(args.use_amp and device.type == "cuda")
        if args.use_amp and not use_amp:
            log("[w] AMP requested but CUDA device not available; running in full precision.")

        scaler_g = amp.GradScaler("cuda", enabled=use_amp)
        scaler_d = amp.GradScaler("cuda", enabled=use_amp)

        if feature_projector is not None:
            trainable_params = list(generator.parameters()) + list(feature_projector.parameters())
        else:
            trainable_params = list(generator.parameters())
        disc_params = list(discriminator.parameters())
        optimizer_g = torch.optim.Adam(trainable_params, lr=2e-4, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(disc_params, lr=1e-4, betas=(0.5, 0.999))
        bce_loss = nn.BCEWithLogitsLoss()

        syn_roots = collect_roots(args.data_syn_dir)
        real_roots = collect_roots(args.data_real_dir)
        syn_t, syn_r = prepare_synthetic_data(syn_roots)
        real_inputs, real_targets = prepare_real_data(real_roots)

        log(f"[i] Loaded {len(syn_t)} synthetic and {len(real_inputs)} real image pairs.")
        if len(real_inputs) > 0:
            log(f"[i] First real sample: {real_inputs[0]}")

        start_epoch = 1
        if args.resume:
            start_epoch = resume_from_checkpoint(
                exp_dir, device, generator,
                discriminator, optimizer_g,
                optimizer_d, args.backbone,
                variant_name,
                feature_projector
            )
            log(f"[i] Resuming training from epoch {start_epoch}")

        validation_images: List[Path] = []
        test_roots = collect_roots(args.test_dir)
        eval_images = prepare_test_images(test_roots)
        if eval_images:
            log(f"[i] Using {len(eval_images)} validation images from --test_dir")
            validation_images = eval_images
        else:
            fallback = real_inputs[:10]
            if fallback:
                log(f"[w] No validation images under {args.test_dir}; using {len(fallback)} real inputs for validation.")
                validation_images = fallback
            else:
                log("[w] No validation images available; per-epoch evaluation disabled.")

        results_root = get_results_dir(args.exp_name)
        if validation_images:
            results_root.mkdir(parents=True, exist_ok=True)

        sigmas = list(np.linspace(1, 5, 80))
        steps_per_epoch = max(len(syn_t) + len(real_inputs), 1)
        global_step = 0

        for epoch in range(start_epoch, args.epochs + 1):
            generator.train()
            generator.feature_extractor.eval()
            discriminator.train()
            if feature_projector is not None:
                feature_projector.train()
            if teacher_generator is not None:
                teacher_generator.eval()
                teacher_generator.feature_extractor.eval()

            epoch_losses: List[float] = []
            epoch_percep: List[float] = []
            epoch_grad: List[float] = []
            epoch_adv: List[float] = []
            epoch_feat_distill: List[float] = []
            epoch_pix_distill: List[float] = []

            step = 0
            attempts = 0
            vis_snapshots: List[Dict[str, np.ndarray]] = []

            while step < steps_per_epoch:
                attempts += 1
                if attempts > steps_per_epoch * 4:
                    break

                use_syn = len(syn_t) > 0 and (len(real_inputs) == 0 or random.random() < 0.7)
                if use_syn:
                    idx = random.randrange(len(syn_t))
                    sample = prepare_synthetic_sample(idx, syn_t, syn_r, sigmas)
                else:
                    if len(real_inputs) == 0:
                        continue
                    idx = random.randrange(len(real_inputs))
                    sample = prepare_real_sample(idx, real_inputs, real_targets)

                if sample is None or not ensure_valid_sample(sample):
                    continue

                input_tensor = image_to_tensor(sample["input"]).unsqueeze(0).to(device=device, dtype=torch.float32)
                target_t_tensor = image_to_tensor(sample["target_t"]).unsqueeze(0).to(device=device, dtype=torch.float32)
                target_r_tensor = image_to_tensor(sample["target_r"]).unsqueeze(0).to(device=device, dtype=torch.float32)

                if step % 2 == 0:
                    optimizer_d.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", enabled=False):
                            fake_t_detach, _ = generator(input_tensor)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pred_real = discriminator(input_tensor, target_t_tensor)
                        pred_fake = discriminator(input_tensor, fake_t_detach)
                        loss_real = bce_loss(pred_real, torch.ones_like(pred_real))
                        loss_fake = bce_loss(pred_fake, torch.zeros_like(pred_fake))
                        d_loss = 0.5 * (loss_real + loss_fake)
                    if not torch.isfinite(d_loss):
                        log(f"[w] Non-finite discriminator loss at epoch {epoch} step {step}; skipping update.")
                        scaler_d.update()
                        continue
                    scaler_d.scale(d_loss).backward()
                    scaler_d.unscale_(optimizer_d)
                    clip_grad_norm_(disc_params, GRAD_CLIP_NORM)
                    scaler_d.step(optimizer_d)
                    scaler_d.update()

                optimizer_g.zero_grad(set_to_none=True)
                teacher_fake_t: Optional[torch.Tensor] = None
                teacher_fake_r: Optional[torch.Tensor] = None
                teacher_feature_maps: Dict[str, torch.Tensor] = {}
                if teacher_generator is not None and (feature_distill_enabled or pixel_distill_enabled):
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", enabled=False):
                            teacher_outputs = teacher_generator(
                                input_tensor, return_features=feature_distill_enabled
                            )
                    if feature_distill_enabled:
                        teacher_fake_t, teacher_fake_r, teacher_feats = teacher_outputs
                        teacher_feature_maps = {
                            name: teacher_feats[name].detach()
                            for name in distill_feature_layers
                            if name in teacher_feats
                        }
                    else:
                        teacher_fake_t, teacher_fake_r = teacher_outputs
                    teacher_fake_t = teacher_fake_t.detach()
                    teacher_fake_r = teacher_fake_r.detach()
                if feature_distill_enabled:
                    fake_t, fake_r, student_features = (
                        generator(input_tensor, return_features=True)
                    )
                else:
                    fake_t, fake_r = generator(input_tensor)
                    student_features = {}
                fake_t = fake_t.float()
                fake_r = fake_r.float()

                with torch.amp.autocast("cuda", enabled=False):
                    percep_t = compute_perceptual_loss(feature_extractor, fake_t, target_t_tensor)
                    if sample["is_syn"]:
                        percep_r = compute_perceptual_loss(feature_extractor, fake_r, target_r_tensor)
                        perceptual_loss = percep_t + percep_r
                        l1_r = compute_l1_loss(fake_r, target_r_tensor)
                        grad_loss = compute_exclusion_loss(fake_t, fake_r, level=3)
                    else:
                        perceptual_loss = percep_t
                        l1_r = torch.zeros(1, device=device, dtype=fake_r.dtype)
                        grad_loss = torch.zeros(1, device=device, dtype=fake_r.dtype)

                    feature_distill_loss = fake_t.new_zeros(1)
                    pixel_distill_loss = fake_t.new_zeros(1)
                    if feature_distill_enabled and teacher_feature_maps:
                        projected_teacher = feature_projector(teacher_feature_maps)
                        feat_losses = []
                        for name in distill_feature_layers:
                            student_map = student_features.get(name)
                            teacher_map = projected_teacher.get(name)
                            if student_map is None or teacher_map is None:
                                continue
                            teacher_map = teacher_map.to(student_map.dtype)
                            feat_losses.append(F.mse_loss(student_map, teacher_map))
                        if feat_losses:
                            feature_distill_loss = torch.stack(feat_losses).mean()
                    if pixel_distill_enabled and teacher_fake_t is not None:
                        pixel_losses = [F.l1_loss(fake_t, teacher_fake_t.to(fake_t.dtype))]
                        if teacher_fake_r is not None and teacher_fake_r.shape == fake_r.shape:
                            pixel_losses.append(F.l1_loss(fake_r, teacher_fake_r.to(fake_r.dtype)))
                        if pixel_losses:
                            pixel_distill_loss = torch.stack(pixel_losses).mean()

                    content_loss = l1_r + 0.2 * perceptual_loss + grad_loss
                    distill_loss = (
                        feature_distill_weight * feature_distill_loss
                        + pixel_distill_weight * pixel_distill_loss
                    )
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_fake_for_g = discriminator(input_tensor, fake_t)
                    adv_loss = bce_loss(pred_fake_for_g, torch.ones_like(pred_fake_for_g))

                adv_loss = adv_loss.float()
                total_g_loss = content_loss * 100.0 + adv_loss + distill_loss
                if not torch.isfinite(total_g_loss):
                    log(f"[w] Non-finite generator loss at epoch {epoch} step {step}; skipping update.")
                    scaler_g.update()
                    continue

                scaler_g.scale(total_g_loss).backward()
                scaler_g.unscale_(optimizer_g)
                clip_grad_norm_(trainable_params, GRAD_CLIP_NORM)
                scaler_g.step(optimizer_g)
                scaler_g.update()

                content_value = float(content_loss.item())
                percep_value = float(perceptual_loss.item())
                grad_value = float(grad_loss.item())
                adv_value = float(adv_loss.item())
                feat_dist_value = float(feature_distill_loss.item()) if feature_distill_enabled else 0.0
                pix_dist_value = float(pixel_distill_loss.item()) if pixel_distill_enabled else 0.0

                epoch_losses.append(content_value)
                epoch_percep.append(percep_value)
                epoch_grad.append(grad_value)
                epoch_adv.append(adv_value)
                if feature_distill_enabled:
                    epoch_feat_distill.append(feat_dist_value)
                if pixel_distill_enabled:
                    epoch_pix_distill.append(pix_dist_value)

                writer.add_scalar("train/content_loss", content_value, global_step)
                writer.add_scalar("train/perceptual_loss", percep_value, global_step)
                writer.add_scalar("train/grad_loss", grad_value, global_step)
                writer.add_scalar("train/adv_loss", adv_value, global_step)
                if feature_distill_enabled:
                    writer.add_scalar("train/feature_distill_loss", feat_dist_value, global_step)
                if pixel_distill_enabled:
                    writer.add_scalar("train/pixel_distill_loss", pix_dist_value, global_step)

                step += 1
                global_step += 1

                if step % max(args.log_interval, 1) == 0:
                    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                    mean_percep = float(np.mean(epoch_percep)) if epoch_percep else 0.0
                    mean_grad = float(np.mean(epoch_grad)) if epoch_grad else 0.0
                    mean_adv = float(np.mean(epoch_adv)) if epoch_adv else 0.0
                    mean_feat = float(np.mean(epoch_feat_distill)) if epoch_feat_distill else 0.0
                    mean_pix = float(np.mean(epoch_pix_distill)) if epoch_pix_distill else 0.0
                    log_msg = (f"[epoch {epoch:03d}] iter {step:04d}/{steps_per_epoch:04d} "
                               f"loss: {mean_loss:.4f} "
                               f"percep: {mean_percep:.4f} "
                               f"grad: {mean_grad:.4f} "
                               f"adv: {mean_adv:.4f}")
                    if feature_distill_enabled:
                        log_msg += f" feat_dist: {mean_feat:.4f}"
                    if pixel_distill_enabled:
                        log_msg += f" pix_dist: {mean_pix:.4f}"
                    log(log_msg)

                fake_t_vis = fake_t.detach().float()
                fake_r_vis = fake_r.detach().float()
                if VISUAL_SNAPSHOTS_PER_EPOCH > 0:
                    snapshot = {
                        "input": np.copy(sample["input"]),
                        "pred_t": tensor_to_image(fake_t_vis[0]),
                        "pred_r": tensor_to_image(fake_r_vis[0]),
                        "file_id": sample["file_id"],
                    }
                    if len(vis_snapshots) < VISUAL_SNAPSHOTS_PER_EPOCH:
                        vis_snapshots.append(snapshot)
                    else:
                        replace_idx = (step - 1) % VISUAL_SNAPSHOTS_PER_EPOCH
                        vis_snapshots[replace_idx] = snapshot

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            mean_percep = float(np.mean(epoch_percep)) if epoch_percep else 0.0
            mean_grad = float(np.mean(epoch_grad)) if epoch_grad else 0.0
            mean_adv = float(np.mean(epoch_adv)) if epoch_adv else 0.0
            mean_feat = float(np.mean(epoch_feat_distill)) if epoch_feat_distill else 0.0
            mean_pix = float(np.mean(epoch_pix_distill)) if epoch_pix_distill else 0.0
            log_msg = (f"[epoch {epoch:03d}] completed "
                       f"| content {mean_loss:.4f} "
                       f"| perceptual {mean_percep:.4f} "
                       f"| grad {mean_grad:.4f} "
                       f"| adv {mean_adv:.4f}")
            if feature_distill_enabled:
                log_msg += f" | feat_dist {mean_feat:.4f}"
            if pixel_distill_enabled:
                log_msg += f" | pix_dist {mean_pix:.4f}"
            log(log_msg)

            writer.add_scalar("train_epoch/content_loss", mean_loss, epoch)
            writer.add_scalar("train_epoch/perceptual_loss", mean_percep, epoch)
            writer.add_scalar("train_epoch/grad_loss", mean_grad, epoch)
            writer.add_scalar("train_epoch/adv_loss", mean_adv, epoch)
            if feature_distill_enabled:
                writer.add_scalar("train_epoch/feature_distill_loss", mean_feat, epoch)
            if pixel_distill_enabled:
                writer.add_scalar("train_epoch/pixel_distill_loss", mean_pix, epoch)

            if args.save_model_freq > 0 and epoch % args.save_model_freq == 0:
                save_checkpoint(exp_dir, epoch, generator, discriminator,
                                optimizer_g, optimizer_d, args.backbone,
                                feature_projector)
                epoch_dir = exp_dir / f"epoch_{epoch:04d}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "backbone": args.backbone,
                    "generator_variant": variant_name,
                    **({"feature_projector": feature_projector.state_dict()} if feature_projector is not None else {}),
                }, epoch_dir / "checkpoint.pt")
                if vis_snapshots:
                    for idx, snapshot in enumerate(vis_snapshots, start=1):
                        save_images(
                            epoch_dir,
                            snapshot["file_id"],
                            snapshot["input"],
                            snapshot["pred_t"],
                            snapshot["pred_r"],
                            index=idx,
                        )
                log(f"[i] Saved checkpoint to {epoch_dir}")
                prune_checkpoints(exp_dir, args.keep_checkpoint_history)

            if validation_images:
                log(f"[i] Running validation inference for epoch {epoch:03d}")
                evaluate_samples(generator, device, validation_images, results_root, epoch, use_amp)
    finally:
        writer.flush()
        writer.close()
        log_file.close()


def inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_root = resolve_path(args.ckpt_dir)
    feature_extractor = create_feature_extractor(
        args.backbone,
        True,
        ckpt_root,
        distributed_hypercolumn=args.use_distributed_hypercolumn,
        hypercolumn_reduction_scale=args.hypercolumn_channel_reduction_scale,
    )
    generator = build_generator(
        feature_extractor,
        residual_skips=args.residual_skips,
        residual_init=args.residual_init,
        output_skip_scale=args.output_skip_scale,
    ).to(device)
    generator.eval()
    generator.feature_extractor.eval()
    variant_name = generator_variant_from_module(generator)

    use_amp = bool(args.use_amp and device.type == "cuda")
    if args.use_amp and not use_amp:
        print("[w] AMP requested but CUDA device not available; running in full precision.")

    exp_dir = get_experiment_dir(args.exp_name)
    load_generator_for_inference(generator, exp_dir, device, args.backbone, expected_variant=variant_name)

    test_roots = collect_roots(args.test_dir)
    images = prepare_test_images(test_roots)
    if not images:
        print(f"[!] No test images found under {args.test_dir}")
        return

    results_root = exp_dir / f"epoch_{0:04d}"
    results_root.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        img = load_image(image_path)
        if img is None:
            continue
        tensor = image_to_tensor(img).unsqueeze(0).to(device)
        start = time.time()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_t, pred_r = generator(tensor)
        elapsed = time.time() - start
        print(f"[i] {image_path} processed in {elapsed:.3f}s")
        pred_t_img = tensor_to_image(pred_t[0])
        pred_r_img = tensor_to_image(pred_r[0])

        subdir = results_root / image_path.stem
        subdir.mkdir(parents=True, exist_ok=True)
        write_rgb_image(subdir / "input.png", img)
        write_rgb_image(subdir / "t_output.png", pred_t_img)
        write_rgb_image(subdir / "r_output.png", pred_r_img)


def main() -> None:
    args = parse_args()
    if args.test_only:
        inference(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

import argparse
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch import amp
from torch.utils.tensorboard import SummaryWriter

from discriminator import PatchDiscriminator
from losses import (compute_exclusion_loss, compute_l1_loss,
                    compute_perceptual_loss)
from models import (DINOFeatureExtractor, FeatureExtractorBase,
                    HypercolumnGenerator, VGGFeatureExtractor)


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp'}
GAUSSIAN_MASK = None
BACKBONE_CHOICES = ["vgg19", "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perceptual reflection removal (PyTorch)")
    parser.add_argument("--exp_name", default="default", help="experiment name (saved under runs/)")
    parser.add_argument("--data_syn_dir", default="", help="comma separated list of synthetic dataset roots")
    parser.add_argument("--data_real_dir", default="", help="comma separated list of real dataset roots")
    parser.add_argument("--save_model_freq", type=int, default=1, help="save model frequency (epochs)")
    parser.add_argument("--keep_checkpoint_history", type=int, default=20, help="number of saved checkpoint epochs to retain (0 keeps all)")
    parser.add_argument("--is_hyper", type=int, default=1, help="use hypercolumn features (1|0)")
    parser.add_argument("--test_only", action="store_true", help="run inference only (skip training)")
    parser.add_argument("--continue_training", action="store_true", help="resume training from the checkpoint stored under runs/--exp_name")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--device", default=None, help="device identifier (e.g. cuda:0)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--log_interval", type=int, default=100, help="iterations between logging updates")
    parser.add_argument("--test_dir", default="./test_images", help="comma separated list of directories for evaluation")
    parser.add_argument("--backbone", default="dinov3_vits16", choices=BACKBONE_CHOICES, help="feature backbone for hypercolumns and perceptual loss")
    parser.add_argument("--ckpt_dir", default="ckpts", help="directory for optional pretrained backbone weights")
    parser.add_argument("--use_amp", action="store_true", help="enable automatic mixed precision during train/test")
    return parser.parse_args()


def get_experiment_dir(exp_name: str) -> Path:
    exp_path = Path(exp_name)
    if exp_path.is_absolute():
        return exp_path
    if exp_path.parts and exp_path.parts[0] == "runs":
        return exp_path
    return Path("runs") / exp_path


def get_results_dir(exp_name: str) -> Path:
    exp_path = Path(exp_name)
    if exp_path.is_absolute():
        exp_path = Path(*exp_path.parts[1:])
    if any(part == ".." for part in exp_path.parts):
        raise ValueError("--exp_name must not contain '..'")
    if not exp_path.parts:
        return Path("test_results") / "default"
    return Path("test_results") / exp_path


def create_feature_extractor(backbone: str, use_hyper: bool, ckpt_dir: Path) -> FeatureExtractorBase:
    name = backbone.lower()
    if name == "vgg19":
        return VGGFeatureExtractor(use_hyper=use_hyper)
    if name in DINOFeatureExtractor.CKPT_FILENAMES:
        return DINOFeatureExtractor(name, use_hyper=use_hyper, ckpt_root=ckpt_dir)
    raise ValueError(f"Unsupported backbone: {backbone}")


def collect_roots(path_string: str) -> List[Path]:
    roots = []
    for fragment in path_string.split(','):
        fragment = fragment.strip()
        if fragment:
            roots.append(Path(fragment))
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
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return np.float32(img) / 255.0


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


def prepare_synthetic_sample(idx: int, transmissions: List[Path], reflections: List[Path], sigmas: List[float]) -> Optional[Dict]:
    syn_t = load_image(transmissions[idx])
    syn_r = load_image(reflections[idx])
    if syn_t is None or syn_r is None:
        return None

    new_w = random.randint(256, 480)
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

    new_w = random.randint(256, 480)
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
                pred_t: np.ndarray, pred_r: np.ndarray) -> None:
    result_dir = epoch_dir / file_id
    result_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(result_dir / "int_t.png"),
                (np.clip(input_img, 0, 1) * 255.0).astype(np.uint8))
    cv2.imwrite(str(result_dir / "out_t.png"), pred_t)
    cv2.imwrite(str(result_dir / "out_r.png"), pred_r)


def save_checkpoint(exp_dir: Path, epoch: int, generator: HypercolumnGenerator,
                    discriminator: PatchDiscriminator, optimizer_g, optimizer_d,
                    backbone: str) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "backbone": backbone,
    }
    torch.save(state, exp_dir / "checkpoint_latest.pt")
    torch.save({
        "state_dict": generator.state_dict(),
        "backbone": backbone,
    }, exp_dir / "generator.pt")


def prune_checkpoints(exp_dir: Path, max_keep: int) -> None:
    if max_keep <= 0:
        return
    checkpoint_dirs = sorted(
        [path for path in exp_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    )
    excess = len(checkpoint_dirs) - max_keep
    for old_dir in checkpoint_dirs[:excess]:
        shutil.rmtree(old_dir, ignore_errors=True)


def resume_from_checkpoint(
    exp_dir: Path, device: torch.device,
    generator: HypercolumnGenerator,
    discriminator: PatchDiscriminator,
    optimizer_g, optimizer_d,
    backbone: str
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
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    return int(checkpoint.get("epoch", 0)) + 1


def load_generator_for_inference(generator: HypercolumnGenerator, exp_dir: Path, device: torch.device, backbone: str) -> None:
    generator_path = exp_dir / "generator.pt"
    if generator_path.exists():
        state = torch.load(generator_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            ckpt_backbone = state.get("backbone", "vgg19")
            if ckpt_backbone != backbone:
                raise ValueError(
                    f"Generator checkpoint backbone ({ckpt_backbone}) does not match requested backbone ({backbone})."
                )
            generator.load_state_dict(state["state_dict"])
        else:
            generator.load_state_dict(state)
        return

    ckpt_path = exp_dir / "checkpoint_latest.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        ckpt_backbone = checkpoint.get("backbone", "vgg19")
        if ckpt_backbone != backbone:
            raise ValueError(
                f"Checkpoint backbone ({ckpt_backbone}) does not match requested backbone ({backbone})."
            )
        generator.load_state_dict(checkpoint["generator"])
        return

    raise FileNotFoundError(f"No generator checkpoint found in {exp_dir}")




def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    use_hyper = args.is_hyper == 1
    ckpt_root = Path(args.ckpt_dir)
    exp_dir = get_experiment_dir(args.exp_name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = exp_dir / "train.log"
    log_file = open(log_path, "a", encoding="utf-8")
    writer = SummaryWriter(log_dir=str(exp_dir))

    def log(message: str) -> None:
        print(message)
        log_file.write(message + "\\n")
        log_file.flush()

    try:
        feature_extractor = create_feature_extractor(args.backbone, use_hyper, ckpt_root)
        generator = HypercolumnGenerator(feature_extractor).to(device)
        discriminator = PatchDiscriminator().to(device)
        feature_extractor = generator.feature_extractor
        feature_extractor.eval()
        log(f"[i] Experiment directory: {exp_dir}")
        log(f"[i] Using backbone: {args.backbone}")

        use_amp = bool(args.use_amp and device.type == "cuda")
        if args.use_amp and not use_amp:
            log("[w] AMP requested but CUDA device not available; running in full precision.")

        scaler_g = amp.GradScaler("cuda", enabled=use_amp)
        scaler_d = amp.GradScaler("cuda", enabled=use_amp)

        optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        bce_loss = nn.BCEWithLogitsLoss()

        syn_roots = collect_roots(args.data_syn_dir)
        real_roots = collect_roots(args.data_real_dir)
        syn_t, syn_r = prepare_synthetic_data(syn_roots)
        real_inputs, real_targets = prepare_real_data(real_roots)

        log(f"[i] Loaded {len(syn_t)} synthetic and {len(real_inputs)} real image pairs.")
        if len(real_inputs) > 0:
            log(f"[i] First real sample: {real_inputs[0]}")

        start_epoch = 1
        if args.continue_training:
            start_epoch = resume_from_checkpoint(
                exp_dir, device, generator,
                discriminator, optimizer_g,
                optimizer_d, args.backbone
            )
            log(f"[i] Resuming training from epoch {start_epoch}")

        sigmas = list(np.linspace(1, 5, 80))
        steps_per_epoch = max(len(syn_t) + len(real_inputs), 1)
        global_step = 0

        for epoch in range(start_epoch, args.epochs + 1):
            generator.train()
            generator.feature_extractor.eval()
            discriminator.train()

            epoch_losses: List[float] = []
            epoch_percep: List[float] = []
            epoch_grad: List[float] = []
            epoch_adv: List[float] = []

            step = 0
            attempts = 0
            last_snapshot = None

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

                input_tensor = image_to_tensor(sample["input"]).unsqueeze(0).to(device)
                target_t_tensor = image_to_tensor(sample["target_t"]).unsqueeze(0).to(device)
                target_r_tensor = image_to_tensor(sample["target_r"]).unsqueeze(0).to(device)

                if step % 2 == 0:
                    optimizer_d.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            fake_t_detach, _ = generator(input_tensor)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pred_real = discriminator(input_tensor, target_t_tensor)
                        pred_fake = discriminator(input_tensor, fake_t_detach)
                        loss_real = bce_loss(pred_real, torch.ones_like(pred_real))
                        loss_fake = bce_loss(pred_fake, torch.zeros_like(pred_fake))
                        d_loss = 0.5 * (loss_real + loss_fake)
                    scaler_d.scale(d_loss).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()

                optimizer_g.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    fake_t, fake_r = generator(input_tensor)
                    pred_fake_for_g = discriminator(input_tensor, fake_t)
                    adv_loss = bce_loss(pred_fake_for_g, torch.ones_like(pred_fake_for_g))

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

                    content_loss = l1_r + 0.2 * perceptual_loss + grad_loss
                    total_g_loss = content_loss * 100.0 + adv_loss

                scaler_g.scale(total_g_loss).backward()
                scaler_g.step(optimizer_g)
                scaler_g.update()

                content_value = float(content_loss.item())
                percep_value = float(perceptual_loss.item())
                grad_value = float(grad_loss.item())
                adv_value = float(adv_loss.item())

                epoch_losses.append(content_value)
                epoch_percep.append(percep_value)
                epoch_grad.append(grad_value)
                epoch_adv.append(adv_value)

                writer.add_scalar("train/content_loss", content_value, global_step)
                writer.add_scalar("train/perceptual_loss", percep_value, global_step)
                writer.add_scalar("train/grad_loss", grad_value, global_step)
                writer.add_scalar("train/adv_loss", adv_value, global_step)

                step += 1
                global_step += 1

                if step % max(args.log_interval, 1) == 0:
                    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                    mean_percep = float(np.mean(epoch_percep)) if epoch_percep else 0.0
                    mean_grad = float(np.mean(epoch_grad)) if epoch_grad else 0.0
                    mean_adv = float(np.mean(epoch_adv)) if epoch_adv else 0.0
                    log(f"[epoch {epoch:03d}] iter {step:04d}/{steps_per_epoch:04d} "
                        f"loss: {mean_loss:.4f} "
                        f"percep: {mean_percep:.4f} "
                        f"grad: {mean_grad:.4f} "
                        f"adv: {mean_adv:.4f}")

                fake_t_vis = fake_t.detach().float()
                fake_r_vis = fake_r.detach().float()
                last_snapshot = {
                    "input": sample["input"],
                    "pred_t": tensor_to_image(fake_t_vis[0]),
                    "pred_r": tensor_to_image(fake_r_vis[0]),
                    "target": tensor_to_image(target_t_tensor[0]),
                    "file_id": sample["file_id"],
                }

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            mean_percep = float(np.mean(epoch_percep)) if epoch_percep else 0.0
            mean_grad = float(np.mean(epoch_grad)) if epoch_grad else 0.0
            mean_adv = float(np.mean(epoch_adv)) if epoch_adv else 0.0
            log(f"[epoch {epoch:03d}] completed "
                f"| content {mean_loss:.4f} "
                f"| perceptual {mean_percep:.4f} "
                f"| grad {mean_grad:.4f} "
                f"| adv {mean_adv:.4f}")

            writer.add_scalar("train_epoch/content_loss", mean_loss, epoch)
            writer.add_scalar("train_epoch/perceptual_loss", mean_percep, epoch)
            writer.add_scalar("train_epoch/grad_loss", mean_grad, epoch)
            writer.add_scalar("train_epoch/adv_loss", mean_adv, epoch)

            if args.save_model_freq > 0 and epoch % args.save_model_freq == 0:
                save_checkpoint(exp_dir, epoch, generator, discriminator,
                                optimizer_g, optimizer_d, args.backbone)
                epoch_dir = exp_dir / f"{epoch:04d}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "backbone": args.backbone,
                }, epoch_dir / "checkpoint.pt")
                if last_snapshot is not None:
                    save_images(epoch_dir, last_snapshot["file_id"],
                                last_snapshot["input"],
                                last_snapshot["pred_t"], last_snapshot["pred_r"])
                log(f"[i] Saved checkpoint to {epoch_dir}")
                prune_checkpoints(exp_dir, args.keep_checkpoint_history)
    finally:
        writer.flush()
        writer.close()
        log_file.close()


def inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_root = Path(args.ckpt_dir)
    feature_extractor = create_feature_extractor(args.backbone,
                                                 args.is_hyper == 1,
                                                 ckpt_root)
    generator = HypercolumnGenerator(feature_extractor).to(device)
    generator.eval()
    generator.feature_extractor.eval()

    use_amp = bool(args.use_amp and device.type == "cuda")
    if args.use_amp and not use_amp:
        print("[w] AMP requested but CUDA device not available; running in full precision.")

    exp_dir = get_experiment_dir(args.exp_name)
    load_generator_for_inference(generator, exp_dir, device, args.backbone)

    test_roots = collect_roots(args.test_dir)
    images = prepare_test_images(test_roots)
    if not images:
        print(f"[!] No test images found under {args.test_dir}")
        return

    results_root = get_results_dir(args.exp_name)
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
        cv2.imwrite(str(subdir / "input.png"),
                    (np.clip(img, 0, 1) * 255.0).astype(np.uint8))
        cv2.imwrite(str(subdir / "t_output.png"), pred_t_img)
        cv2.imwrite(str(subdir / "r_output.png"), pred_r_img)


def main() -> None:
    args = parse_args()
    if args.test_only:
        inference(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

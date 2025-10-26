from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.hub
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19


def identity_init(conv: nn.Conv2d) -> None:
    """Initialise conv kernels as (approximate) identity."""
    if not isinstance(conv, nn.Conv2d):
        return

    weight = conv.weight.data
    weight.zero_()
    out_channels, in_channels, kh, kw = weight.shape
    cx, cy = kh // 2, kw // 2
    dim = min(in_channels, out_channels)
    for i in range(dim):
        weight[i, i, cx, cy] = 1.0
    if conv.bias is not None:
        conv.bias.data.zero_()


class NormalizedMix(nn.Module):
    """Learns a mixture between identity and BatchNorm outputs."""

    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.w0 = nn.Parameter(torch.ones(1))
        self.w1 = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w0 * x + self.w1 * self.bn(x)


class ConvNormActivation(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = dilation if kernel_size > 1 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=True)
        self.norm = NormalizedMix(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        identity_init(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class FeatureExtractorBase(nn.Module):
    """Base interface for perceptual backbones."""

    def __init__(self, use_hyper: bool = True):
        super().__init__()
        self.use_hyper = use_hyper

    @property
    def hyper_layers(self) -> List[str]:
        raise NotImplementedError

    @property
    def perceptual_layers(self) -> List[Tuple[str, float]]:
        raise NotImplementedError

    @property
    def hypercolumn_channels(self) -> int:
        raise NotImplementedError

    def extract_features(self, x: torch.Tensor, require_grad: bool = True) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def build_hypercolumns(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate hypercolumn features with the original input."""
        features = self.extract_features(x, require_grad=True)
        hyper_maps: List[torch.Tensor] = []
        for name in self.hyper_layers:
            feat = features[name]
            if feat.shape[-2:] != x.shape[-2:]:
                feat = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
            if self.use_hyper:
                hyper_maps.append(feat)
            else:
                hyper_maps.append(torch.zeros_like(feat))
        return torch.cat(hyper_maps + [x], dim=1)


class HypercolumnGenerator(nn.Module):
    """Generator with configurable hypercolumn feature extractor."""

    def __init__(self, feature_extractor: FeatureExtractorBase, base_channels: int = 64):
        super().__init__()
        self.feature_extractor = feature_extractor
        in_channels = 3 + feature_extractor.hypercolumn_channels

        self.conv0 = ConvNormActivation(in_channels, base_channels, kernel_size=1)
        self.conv1 = ConvNormActivation(base_channels, base_channels, 3, 1)
        self.conv2 = ConvNormActivation(base_channels, base_channels, 3, 2)
        self.conv3 = ConvNormActivation(base_channels, base_channels, 3, 4)
        self.conv4 = ConvNormActivation(base_channels, base_channels, 3, 8)
        self.conv5 = ConvNormActivation(base_channels, base_channels, 3, 16)
        self.conv6 = ConvNormActivation(base_channels, base_channels, 3, 32)
        self.conv7 = ConvNormActivation(base_channels, base_channels, 3, 64)
        self.conv9 = ConvNormActivation(base_channels, base_channels, 3, 1)
        self.conv_last = nn.Conv2d(base_channels, 6, kernel_size=1)
        nn.init.kaiming_normal_(self.conv_last.weight, a=0.2,
                                nonlinearity="leaky_relu")
        nn.init.zeros_(self.conv_last.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hyper_input = self.feature_extractor.build_hypercolumns(x)
        net = self.conv0(hyper_input)
        net = self.conv1(net)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.conv5(net)
        net = self.conv6(net)
        net = self.conv7(net)
        net = self.conv9(net)
        outputs = self.conv_last(net)
        transmission, reflection = torch.chunk(outputs, 2, dim=1)
        return transmission, reflection


class VGGFeatureExtractor(FeatureExtractorBase):
    """VGG19-based hypercolumn and perceptual feature extractor."""

    _HYPER_LAYERS = ["conv1_2", "conv2_2", "conv3_2", "conv4_2", "conv5_2"]
    _LAYER_DIMS = {
        "conv1_2": 64,
        "conv2_2": 128,
        "conv3_2": 256,
        "conv4_2": 512,
        "conv5_2": 512,
        "input": 3,
    }
    _PERCEPTUAL_LAYERS = [
        ("input", 1.0),
        ("conv1_2", 1.0 / 2.6),
        ("conv2_2", 1.0 / 4.8),
        ("conv3_2", 1.0 / 3.7),
        ("conv4_2", 1.0 / 5.6),
        ("conv5_2", 10.0 / 1.5),
    ]

    def __init__(self, use_hyper: bool = True, weights: VGG19_Weights = VGG19_Weights.IMAGENET1K_V1):
        super().__init__(use_hyper=use_hyper)
        vgg = vgg19(weights=weights).features
        self.slice1 = vgg[:4]
        self.slice2 = vgg[4:9]
        self.slice3 = vgg[9:18]
        self.slice4 = vgg[18:27]
        self.slice5 = vgg[27:36]

        for param in self.parameters():
            param.requires_grad = False

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @property
    def hyper_layers(self) -> List[str]:
        return self._HYPER_LAYERS

    @property
    def perceptual_layers(self) -> List[Tuple[str, float]]:
        return self._PERCEPTUAL_LAYERS

    @property
    def hypercolumn_channels(self) -> int:
        return sum(self._LAYER_DIMS[layer] for layer in self._HYPER_LAYERS)

    def extract_features(self, x: torch.Tensor, require_grad: bool = True) -> Dict[str, torch.Tensor]:
        x_norm = (x - self.mean) / self.std
        if not require_grad:
            x_norm = x_norm.detach()

        features: Dict[str, torch.Tensor] = {"input": x_norm}
        h = self.slice1(x_norm)
        features["conv1_2"] = h
        h = self.slice2(h)
        features["conv2_2"] = h
        h = self.slice3(h)
        features["conv3_2"] = h
        h = self.slice4(h)
        features["conv4_2"] = h
        h = self.slice5(h)
        features["conv5_2"] = h
        return features


class DINOFeatureExtractor(FeatureExtractorBase):
    """DINOv3 ViT-based feature extractor with torch.hub support."""

    CKPT_FILENAMES = {
        "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    }
    DEFAULT_REPO = "facebookresearch/dinov3"
    DEFAULT_REF = "7bf81b2a0eb0e330dbc84a5d3d31d86ed3cdbd84"

    def __init__(self, arch: str, use_hyper: bool = True, ckpt_root: Optional[Path] = None):
        super().__init__(use_hyper=use_hyper)
        self.arch = arch
        self.ckpt_root = Path(ckpt_root) if ckpt_root is not None else None
        self.weights_path = self._resolve_weights_path()
        self.model = self._load_model()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        patch_size = getattr(self.model, "patch_size", None)
        if patch_size is None:
            patch_embed = getattr(self.model, "patch_embed", None)
            patch_size = getattr(patch_embed, "patch_size", 16)
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)

        self.embed_dim = getattr(self.model, "embed_dim", None)
        if self.embed_dim is None:
            raise ValueError(f"Unable to determine embed_dim for {arch}")

        num_blocks = len(getattr(self.model, "blocks", []))
        if num_blocks == 0:
            raise ValueError(f"No transformer blocks found in {arch}")
        self.layer_indices = self._select_layers(num_blocks)
        self._hyper_layers = [f"block_{idx}" for idx in self.layer_indices]
        self._layer_dims = {layer: self.embed_dim for layer in self._hyper_layers}
        self._layer_dims["final"] = self.embed_dim
        self._layer_dims["input"] = 3

        self._perceptual_layers: List[Tuple[str, float]] = [
            ("input", 1.0),
            *[(layer, 1.0) for layer in self._hyper_layers],
            ("final", 1.0),
        ]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self._feature_cache: Dict[str, torch.Tensor] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _resolve_weights_path(self) -> Optional[str]:
        if self.ckpt_root is None:
            return None
        filename = self.CKPT_FILENAMES.get(self.arch)
        if not filename:
            return None
        path = self.ckpt_root / filename
        return str(path) if path.is_file() else None

    def _load_model(self) -> nn.Module:
        repo = os.environ.get("DINOV3_HUB_REPO", self.DEFAULT_REPO)
        pinned_ref = os.environ.get("DINOV3_HUB_REF", self.DEFAULT_REF)
        fallback_ref = os.environ.get("DINOV3_HUB_FALLBACK_REF", "main")

        def _format_repo(r: str, ref: Optional[str]) -> str:
            return f"{r}:{ref}" if ref else r

        repo_spec = _format_repo(repo, pinned_ref)
        fallback_spec = _format_repo(repo, fallback_ref) if fallback_ref else None

        def _load(repo_or_dir: str) -> nn.Module:
            hub_kwargs = {
                "repo_or_dir": repo_or_dir,
                "model": self.arch,
                "source": "github",
                "trust_repo": True,
            }
            if self.weights_path:
                hub_kwargs["weights"] = self.weights_path
            return torch.hub.load(**hub_kwargs)

        try:
            return _load(repo_spec)
        except Exception as error:
            missing_commit = (
                pinned_ref
                and isinstance(error, ValueError)
                and "Cannot find" in str(error)
                and repo in str(error)
            )
            can_fallback = missing_commit and fallback_spec and fallback_spec != repo_spec
            if can_fallback:
                warnings.warn(
                    (
                        f"Falling back to '{fallback_spec}' because "
                        f"DINOv3 commit '{pinned_ref}' is not available upstream. "
                        "Set DINOV3_HUB_REF to override."
                    ),
                    RuntimeWarning,
                )
                try:
                    return _load(fallback_spec)
                except Exception:
                    raise error
            raise

    def _select_layers(self, total_blocks: int) -> List[int]:
        if total_blocks <= 4:
            return list(range(total_blocks))
        step = total_blocks - 1
        # Sample four layers evenly across depth.
        indices = torch.linspace(0, step, steps=4)
        return sorted({int(round(idx.item())) for idx in indices})

    def _register_hooks(self) -> None:
        def make_hook(name: str):
            def hook(_: nn.Module, __: Tuple[torch.Tensor, ...], output: torch.Tensor):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                self._feature_cache[name] = tensor

            return hook

        for idx, name in zip(self.layer_indices, self._hyper_layers):
            handle = self.model.blocks[idx].register_forward_hook(make_hook(name))
            self._hook_handles.append(handle)

    @property
    def hyper_layers(self) -> List[str]:
        return self._hyper_layers

    @property
    def perceptual_layers(self) -> List[Tuple[str, float]]:
        return self._perceptual_layers

    @property
    def hypercolumn_channels(self) -> int:
        return sum(self._layer_dims[layer] for layer in self._hyper_layers)

    def extract_features(self, x: torch.Tensor, require_grad: bool = True) -> Dict[str, torch.Tensor]:
        self._feature_cache = {}
        x_norm = (x - self.mean) / self.std
        if not require_grad:
            x_norm = x_norm.detach()

        padded, pad_h, pad_w = self._pad_input(x_norm)
        forward_ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with forward_ctx:
            outputs = self.model.forward_features(padded)
        tokens = outputs.get("x_norm_patchtokens")
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        if tokens is None:
            tokens = outputs.get("patch_tokens")
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        if tokens is None:
            raise ValueError(f"{self.arch} did not return patch tokens.")
        self._feature_cache["final"] = tokens

        maps: Dict[str, torch.Tensor] = {}
        full_h, full_w = padded.shape[-2:]
        target_h = full_h - pad_h
        target_w = full_w - pad_w

        required_layers = {name for name, _ in self._perceptual_layers}
        required_layers.update(self._hyper_layers)
        required_layers.add("final")

        for name in required_layers:
            tensor = self._feature_cache.get(name)
            if tensor is None and name == "final":
                tensor = tokens
            if tensor is None:
                continue
            maps[name] = self._tokens_to_map(tensor, full_h, full_w, target_h, target_w)

        maps["input"] = x_norm
        self._feature_cache = {}
        return maps

    def _pad_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        h, w = x.shape[-2:]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return x, 0, 0
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return padded, pad_h, pad_w

    def _tokens_to_map(self, tokens: torch.Tensor, padded_h: int, padded_w: int, target_h: int, target_w: int) -> torch.Tensor:
        tensor = tokens[0] if isinstance(tokens, (tuple, list)) else tokens
        if tensor.dim() == 3:
            grid_h = padded_h // self.patch_size
            grid_w = padded_w // self.patch_size
            expected = grid_h * grid_w
            num_tokens = tensor.shape[1]
            if num_tokens < expected:
                raise ValueError("Token count does not match spatial resolution.")
            if num_tokens > expected:
                tensor = tensor[:, num_tokens - expected :, :]
            b, _, dim = tensor.shape
            tensor = tensor[:, :expected, :]
            tensor = tensor.transpose(1, 2).reshape(b, dim, grid_h, grid_w)
            feat_map = tensor
        elif tensor.dim() == 4:
            feat_map = tensor
        else:
            raise ValueError("Unsupported tensor shape for feature conversion.")

        if feat_map.shape[-2:] != (padded_h, padded_w):
            feat_map = F.interpolate(feat_map, size=(padded_h, padded_w), mode="bilinear", align_corners=False)
        if padded_h != target_h or padded_w != target_w:
            feat_map = feat_map[:, :, :target_h, :target_w]
        return feat_map

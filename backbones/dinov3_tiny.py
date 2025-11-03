"""
Minimal DINOv3 ViT-Tiny backbone extracted from the DEIMv2 project.
This implements the architecture used to finetune the custom checkpoint
stored under ``ckpts/deimv2_dinov3_s_wholebody34.pth``.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Iterable, Literal, Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension for rotary position embeddings."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to queries/keys."""
    return (x * cos) + (rotate_half(x) * sin)


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim % 4 != 0:
            raise ValueError("Head dimension must be divisible by 4 for 2D RoPE.")
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        self.register_buffer(
            "periods",
            torch.empty(head_dim // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        dd = {"device": device, "dtype": dtype}

        if self.normalize_coords == "max":
            max_hw = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_hw
            coords_w = torch.arange(0.5, W, **dd) / max_hw
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            min_hw = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_hw
            coords_w = torch.arange(0.5, W, **dd) / min_hw

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            coords += torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)[None, :]
        if self.training and self.jitter_coords is not None:
            jitter = torch.empty(2, **dd).uniform_(
                -math.log(self.jitter_coords),
                math.log(self.jitter_coords),
            )
            coords *= jitter.exp()[None, :]
        if self.training and self.rescale_coords is not None:
            rescale = torch.empty(1, **dd).uniform_(
                -math.log(self.rescale_coords),
                math.log(self.rescale_coords),
            )
            coords *= rescale.exp()

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).repeat(1, 2)

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)

    def _init_weights(self) -> None:
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        if self.base is not None:
            exponents = torch.arange(self.periods.numel(), device=device, dtype=dtype)
            periods = self.base ** (2 * exponents / (self.periods.numel() * 2))
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.periods.numel(), device=device, dtype=dtype)
            periods = self.max_period * (base ** (exponents - 1))
        self.periods.data.copy_(periods)


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob or 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(
        self,
        x: torch.Tensor,
        rope_sincos: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if rope_sincos is not None:
            sin, cos = rope_sincos
            q_cls, q_patch = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_patch = k[:, :, :1, :], k[:, :, 1:, :]
            q_patch = apply_rope(q_patch, sin, cos)
            k_patch = apply_rope(k_patch, sin, cos)
            q = torch.cat((q_cls, q_patch), dim=2)
            k = torch.cat((k_cls, k_patch), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        rope_sincos: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), rope_sincos=rope_sincos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 192) -> None:
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    def norm_cdf(x: torch.Tensor | float) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=tensor.dtype, device=tensor.device)
        return (1.0 + torch.erf(x / math.sqrt(2.0))) / 2.0

    if mean < a - 2 * std or mean > b + 2 * std:
        raise ValueError("mean is more than 2 std from [a, b] in trunc_normal_.")

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


class VisionTransformer(nn.Module):
    """ViT-Tiny backbone with rotary embeddings matching the DEIMv2 checkpoint."""

    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        return_layers: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        self.return_layers = set(return_layers or [])
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size

        self._model = nn.Module()
        self._model.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self._model.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self._model.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop_path=float(i) / depth * 0.0,
            )
            for i in range(depth)
        )
        self._model.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100.0,
            normalize_coords="separate",
            shift_coords=None,
            jitter_coords=None,
            rescale_coords=None,
            dtype=None,
            device=None,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self._model.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return intermediate token tensors for compatibility with DEIMv2."""
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

        x_embed = self._model.patch_embed(x)
        cls_token = self._model.cls_token.expand(x_embed.shape[0], -1, -1)
        x_tokens = torch.cat((cls_token, x_embed), dim=1)

        patch_grid_h = x.shape[2] // self.patch_size
        patch_grid_w = x.shape[3] // self.patch_size
        rope_sincos = self._model.rope_embed(H=patch_grid_h, W=patch_grid_w)

        for idx, block in enumerate(self._model.blocks):
            x_tokens = block(x_tokens, rope_sincos=rope_sincos)
            if idx in self.return_layers:
                outputs.append((x_tokens[:, 1:], x_tokens[:, :1]))

        if not outputs:
            outputs.append((x_tokens[:, 1:], x_tokens[:, :1]))
        return outputs

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return patch tokens in the same dict format as torchvision DINOv3."""
        B, _, H, W = x.shape
        x_embed = self._model.patch_embed(x)
        cls_token = self._model.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls_token, x_embed), dim=1)

        patch_grid_h = H // self.patch_size
        patch_grid_w = W // self.patch_size
        rope_sincos = self._model.rope_embed(H=patch_grid_h, W=patch_grid_w)

        for block in self._model.blocks:
            x_tokens = block(x_tokens, rope_sincos=rope_sincos)

        patch_tokens = x_tokens[:, 1:, :]
        return {
            "patch_tokens": patch_tokens,
            "x_norm_patchtokens": patch_tokens,
        }
    @property
    def blocks(self) -> nn.ModuleList:
        return self._model.blocks

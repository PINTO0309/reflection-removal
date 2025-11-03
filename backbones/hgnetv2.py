"""
Simplified HGNetV2 backbone extracted from the DEIMv2 project.
This definition is sufficient to load the custom checkpoint stored in
``ckpts/deimv2_hgnetv2_n_wholebody34.pth``.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HGNetV2"]


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        padding: str | int = "",
        use_act: bool = True,
        use_lab: bool = False,
        act: str = "relu",
    ) -> None:
        super().__init__()
        if padding == "same":
            conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(in_chs, out_chs, kernel_size, stride, groups=groups, bias=False),
            )
        else:
            pad = (kernel_size - 1) // 2 if isinstance(kernel_size, int) else 0
            conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding=pad, groups=groups, bias=False)

        self.conv = conv
        self.bn = nn.BatchNorm2d(out_chs)
        self.use_act = use_act
        self.use_lab = use_lab
        self.act = self._make_act(act) if use_act else nn.Identity()
        self.lab = LearnableAffineBlock() if (use_act and use_lab) else nn.Identity()

    @staticmethod
    def _make_act(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.lab(x)


class LightConvBNAct(nn.Module):
    def __init__(self, in_chs: int, out_chs: int, kernel_size: int, use_lab: bool = False, act: str = "relu") -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_chs, out_chs, kernel_size=1, use_act=False, use_lab=use_lab, act=act)
        self.conv2 = ConvBNAct(out_chs, out_chs, kernel_size=kernel_size, groups=out_chs, use_lab=use_lab, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    def __init__(self, in_chs: int, mid_chs: int, out_chs: int, use_lab: bool = False, act: str = "relu") -> None:
        super().__init__()
        self.stem1 = ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab, act=act)
        self.stem2a = ConvBNAct(mid_chs, mid_chs // 2, kernel_size=2, stride=1, use_lab=use_lab, act=act)
        self.stem2b = ConvBNAct(mid_chs // 2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab, act=act)
        self.stem3 = ConvBNAct(mid_chs * 2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab, act=act)
        self.stem4 = ConvBNAct(mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act=act)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(chs, chs, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return identity * x


class HG_Block(nn.Module):
    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        layer_num: int,
        *,
        kernel_size: int = 3,
        residual: bool = False,
        light_block: bool = False,
        use_lab: bool = False,
        agg: str = "se",
        act: str = "relu",
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(layer_num):
            block_cls = LightConvBNAct if light_block else ConvBNAct
            layers.append(
                block_cls(
                    in_chs if i == 0 else mid_chs,
                    mid_chs,
                    kernel_size=kernel_size,
                    use_lab=use_lab,
                    act=act,
                )
            )
        self.layers = nn.ModuleList(layers)
        total_chs = in_chs + layer_num * mid_chs
        if agg == "se":
            self.aggregation = nn.Sequential(
                ConvBNAct(total_chs, out_chs // 2, kernel_size=1, stride=1, use_lab=use_lab, act=act),
                ConvBNAct(out_chs // 2, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act=act),
            )
        elif agg == "ese":
            self.aggregation = nn.Sequential(
                ConvBNAct(total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act=act),
                EseModule(out_chs),
            )
        else:
            raise ValueError(f"Unsupported aggregation type: {agg}")
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        outputs = [x]
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = torch.cat(outputs, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = x + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        block_num: int,
        layer_num: int,
        *,
        downsample: bool,
        light_block: bool,
        kernel_size: int,
        agg: str = "se",
        use_lab: bool,
        act: str,
    ) -> None:
        super().__init__()
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
                act=act,
            )
        else:
            self.downsample = nn.Identity()

        blocks: List[nn.Module] = []
        for i in range(block_num):
            blocks.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=(i != 0),
                    kernel_size=kernel_size,
                    light_block=light_block,
                    agg=agg,
                    use_lab=use_lab,
                    act=act,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class HGNetV2(nn.Module):
    """HGNetV2 backbone variant used for the DEIMv2 checkpoints."""

    arch_configs = {
        "B0": {
            "stem_channels": [3, 16, 16],
            "stage_config": {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
        },
    }

    def __init__(
        self,
        name: str = "B0",
        *,
        use_lab: bool = True,
        return_indices: Iterable[int] = (0, 1, 2, 3),
        act: str = "relu",
    ) -> None:
        super().__init__()
        if name not in self.arch_configs:
            raise ValueError(f"Unsupported HGNetV2 variant: {name}")
        config = self.arch_configs[name]
        self.return_indices = list(return_indices)

        stem_channels = config["stem_channels"]
        stage_config = config["stage_config"]

        self.stem = StemBlock(
            in_chs=stem_channels[0],
            mid_chs=stem_channels[1],
            out_chs=stem_channels[2],
            use_lab=use_lab,
            act=act,
        )

        self.stages = nn.ModuleList()
        for stage_name in stage_config:
            in_chs, mid_chs, out_chs, block_num, downsample, light_block, kernel_size, layer_num = stage_config[stage_name]
            stage = HG_Stage(
                in_chs,
                mid_chs,
                out_chs,
                block_num,
                layer_num,
                downsample=downsample,
                light_block=light_block,
                kernel_size=kernel_size,
                agg="se",
                use_lab=use_lab,
                act=act,
            )
            self.stages.append(stage)

        self.out_channels = [stage_config[k][2] for k in stage_config]
        self.out_strides = [4, 8, 16, 32][: len(self.out_channels)]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outputs: list[torch.Tensor] = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_indices:
                outputs.append(x)
        return outputs

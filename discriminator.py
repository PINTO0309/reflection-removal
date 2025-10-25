import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator replicating the TensorFlow architecture."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = self._make_block(base_channels, base_channels * 2, stride=2)
        self.layer3 = self._make_block(base_channels * 2, base_channels * 4, stride=2)
        self.layer4 = self._make_block(base_channels * 4, min(base_channels * 8, 512), stride=1)
        self.layer5 = nn.Conv2d(min(base_channels * 8, 512), 1, kernel_size=4, stride=1, padding=1)

        self._init_weights()

    @staticmethod
    def _make_block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Input image tensor (B, C, H, W)
            targets: Target / generated tensor (B, C, H, W)
        Returns:
            Patch-level logits (B, 1, H', W')
        """
        x = torch.cat([inputs, targets], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

"""The three networks used by DLL-GAN.

Generator          - ResNet with 9 residual blocks (CycleGAN-style), as the paper.
PixelDiscriminator - 1x1 PatchGAN: classifies each pixel real/fake.
Regressor          - ResNet-18 with a single regression output = degradation level.
"""
import torch
import torch.nn as nn
import torchvision


def _norm(norm, c):
    # The paper specifies batch normalization; InstanceNorm often trains more
    # stably for image generation, so it is exposed as an option.
    return nn.BatchNorm2d(c) if norm == "batch" else nn.InstanceNorm2d(c)


class ResidualBlock(nn.Module):
    def __init__(self, c, norm="batch"):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3),
            _norm(norm, c),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3),
            _norm(norm, c),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """ResNet generator: 7x7 stem, 2x downsampling, 9 residual blocks, 2x
    upsampling (transposed conv), 7x7 head + Tanh. Output size == input size
    (input H, W must be divisible by 4)."""

    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9, norm="batch"):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7),
            _norm(norm, ngf),
            nn.ReLU(True),
        ]
        c = ngf
        for _ in range(2):  # downsampling
            layers += [nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                       _norm(norm, c * 2), nn.ReLU(True)]
            c *= 2
        for _ in range(n_blocks):  # residual blocks
            layers += [ResidualBlock(c, norm)]
        for _ in range(2):  # upsampling
            layers += [nn.ConvTranspose2d(c, c // 2, 3, stride=2, padding=1,
                                          output_padding=1),
                       _norm(norm, c // 2), nn.ReLU(True)]
            c //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(c, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PixelDiscriminator(nn.Module):
    """1x1 PatchGAN. Returns per-pixel logits (no sigmoid; use BCEWithLogits)."""

    def __init__(self, in_ch=3, ndf=64, norm="batch"):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 1),
            _norm(norm, ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, 1),
        )

    def forward(self, x):
        return self.model(x)


class Regressor(nn.Module):
    """ResNet-18 whose classifier head is replaced by a single scalar output
    estimating the degradation level (0 = clean)."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, 1)
        self.net = net

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B,)

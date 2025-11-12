from typing import Literal, Tuple

from torch import Tensor, nn


class ResNetBlock(nn.Module):
    def __init__(self, num_channels: int, num_groups: int):
        super(ResNetBlock, self).__init__()
        assert num_channels % num_groups == 0

        self.module = nn.Sequential(
            nn.GroupNorm(num_groups, num_channels, eps=1e-6),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)


class ResNet(nn.Module):
    def __init__(
        self,
        image_channels: int,
        dims: Tuple[int, ...],
        depths: Tuple[int, ...],
        sample: Literal["up", "down"],
    ):
        super(ResNet, self).__init__()

        assert len(dims) == len(depths)

        layers = [nn.Conv2d(image_channels, dims[0], kernel_size=5, padding=2)]
        for i in range(len(dims) - 1):
            layers += [ResNetBlock(dims[i], num_groups=32) for _ in range(depths[i])]

            if sample == "up":
                layers += [
                    nn.ConvTranspose2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
                ]
            else:
                layers += [nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)]

        layers += [ResNetBlock(dims[-1], num_groups=32) for _ in range(depths[-1])]

        self.module = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

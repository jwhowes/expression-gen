from typing import Tuple

import torch
from torch import nn, Tensor
from dataclasses import dataclass

from src.config import Config
from .resnet import ResNet


@dataclass
class DiagonalGaussian:
    mean: Tensor
    log_var: Tensor

    def sample(self) -> Tensor:
        return torch.randn_like(self.log_var) * self.log_var.exp() + self.mean

    def kl(self) -> Tensor:
        return 0.5 * (self.mean.pow(2) + self.log_var.exp() - 1.0 - self.log_var).sum(
            (1, 2, 3)
        )


class VAEEncoder(ResNet):
    def __init__(
        self,
        image_channels: int,
        latent_channels: int,
        dims: Tuple[int, ...],
        depths: Tuple[int, ...],
    ):
        super(VAEEncoder, self).__init__(image_channels, dims, depths, sample="down")

        self.head = nn.Conv2d(dims[-1], 2 * latent_channels, kernel_size=1)

    def forward(self, x: Tensor) -> DiagonalGaussian:
        mean, log_var = self.head(super().forward(x)).chunk(2, 1)

        return DiagonalGaussian(mean=mean, log_var=log_var)


class VAEDecoder(ResNet):
    def __init__(
        self,
        image_channels: int,
        latent_channels: int,
        dims: Tuple[int, ...],
        depths: Tuple[int, ...],
    ):
        super(VAEDecoder, self).__init__(latent_channels, dims, depths, sample="up")

        self.head = nn.Conv2d(dims[-1], image_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(super().forward(x))


class ResNetConfig(Config):
    dims: Tuple[int, ...]
    depths: Tuple[int, ...]


class VAEConfig(Config):
    image_channels: int
    latent_channels: int

    encoder: ResNetConfig
    decoder: ResNetConfig

    def get_encoder(self) -> VAEEncoder:
        return VAEEncoder(
            image_channels=self.image_channels,
            latent_channels=self.latent_channels,
            **self.encoder.model_dump(),
        )

    def get_decoder(self) -> VAEDecoder:
        return VAEDecoder(
            image_channels=self.image_channels,
            latent_channels=self.latent_channels,
            **self.encoder.model_dump(),
        )

    def get_vae(self) -> Tuple[VAEEncoder, VAEDecoder]:
        return self.get_encoder(), self.get_decoder()

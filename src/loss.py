from typing import Tuple

import torch.nn.functional as F

from torch import nn, Tensor

from src.model.vae import DiagonalGaussian
from src.config import Config


class VAELossConfig(Config):
    kl_weight: float


class VAELoss(nn.Module):
    def __init__(self, kl_weight: float):
        super(VAELoss, self).__init__()

        self.kl_weight = kl_weight

    def forward(
        self, recon: Tensor, image: Tensor, dist: DiagonalGaussian
    ) -> Tuple[Tensor, Tensor, Tensor]:
        recon_loss = F.mse_loss(recon, image)
        kl = dist.kl().mean()

        return (
            recon_loss + self.kl_weight * dist.kl().mean(),
            recon_loss.item(),
            kl.item(),
        )

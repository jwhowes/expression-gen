import torch.nn.functional as F

from torch import nn, Tensor

from src.model.vae import DiagonalGaussian


class VAELoss(nn.Module):
    def __init__(self, kl_weight: float):
        super(VAELoss, self).__init__()

        self.kl_weight = kl_weight

    def forward(self, pred: Tensor, recon: Tensor, dist: DiagonalGaussian) -> Tensor:
        return F.mse_loss(pred, recon) + self.kl_weight * dist.kl().mean()

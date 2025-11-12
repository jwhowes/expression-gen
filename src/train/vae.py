import torch

from typing import Optional

from torch.utils.data import DataLoader
from src.model.vae import DiagonalGaussian, VAEConfig
from src.config import Config
from src.data import ExpressionDatasetConfig
from src.loss import VAELossConfig, VAELoss
from .trainer import Trainer


class VAETrainerConfig(Config):
    num_epochs: int
    log_interval: int

    lr: float
    weight_decay: float
    clip_grad: Optional[float] = None

    model: VAEConfig

    loss: VAELossConfig

    dataset: ExpressionDatasetConfig


class VAETrainer(Trainer):
    config: VAETrainerConfig
    metrics = ("recon_loss", "kl")
    config_type = VAETrainerConfig

    def train(self):
        encoder, decoder = self.config.model.get_vae()

        criterion = VAELoss(kl_weight=self.config.loss.kl_weight)

        dataset = self.config.dataset.get_dataset()
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=True,
        )

        opt = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch + 1} / {self.config.num_epochs}")
            for i, (image, _) in enumerate(dataloader):
                opt.zero_grad()

                dist: DiagonalGaussian = encoder(image)

                z = dist.sample()
                recon = decoder(z)

                loss, recon_loss, kl = criterion(recon, image, dist)

                loss.backward()
                if self.config.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(decoder.parameters()),
                        self.config.clip_grad,
                    )

                opt.step()

                if i % self.config.log_interval == 0:
                    print(
                        f"\tStep {i} / {len(dataloader)}.\tRecon: {recon_loss:.4f}\tKL: {kl:.4f}"
                    )

                    self.log(
                        {
                            "encoder": encoder.state_dict(),
                            "decoder": decoder.state_dict(),
                        },
                        recon_loss=recon_loss,
                        kl=kl,
                    )

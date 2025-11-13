import torch

from typing import Optional
from pydantic import Field
from torch import Tensor
from torch.utils.data import DataLoader

from src.config import Config
from src.model.flow import FlowMatchModelConfig
from src.data import ExpressionDatasetConfig
from src.train.trainer import Trainer
from src.loss import FlowMatchLoss
from src.model.vae import VAEEncoder


class SamplerConfig(Config):
    mean: float = Field(default=-0.5)
    std: float = Field(default=1.0)

    sigma_min: float = Field(default=1e-4)


class EncoderConfig(Config):
    exp_name: str
    ckpt: int


class FlowMatchConfig(Config):
    sampler: SamplerConfig

    num_epochs: int
    log_interval: int

    lr: float
    weight_decay: float
    clip_grad: Optional[float] = None

    model: FlowMatchModelConfig

    encoder: EncoderConfig

    dataset: ExpressionDatasetConfig


class FlowMatchTrainer(Trainer):
    config_type = FlowMatchConfig
    config: FlowMatchConfig

    metrics = ("loss",)

    def load_encoder(self) -> VAEEncoder:
        raise NotImplementedError

    def train(self):
        encoder = self.load_encoder()

        model = self.config.model.get_model()

        dataset = self.config.dataset.get_dataset()
        dataloader = DataLoader(
            dataset, batch_size=self.config.dataset.batch_size, shuffle=True
        )

        opt = torch.optim.AdamW(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        criterion = FlowMatchLoss(self.config.sampler.sigma_min)

        sigma_offset = 1 - self.config.sampler.sigma_min

        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch + 1} / {self.config.num_epochs}")
            for i, (image, label) in enumerate(dataloader):
                B = image.shape[0]

                with torch.no_grad():
                    x_0: Tensor = encoder(image).sample()

                opt.zero_grad()

                t = (
                    torch.randn(B) * self.config.sampler.std + self.config.sampler.mean
                ).sigmoid()

                x_1 = torch.randn_like(image)

                x_t = (1 - sigma_offset * t.view(B, 1, 1, 1)) * x_0 + t.view(
                    B, 1, 1, 1
                ) * x_1

                pred_flow = model(x_t, label, t)

                loss = criterion(pred_flow, x_0, x_1)
                loss.backward()

                if self.config.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.clip_grad
                    )

                opt.step()

                if i % self.config.log_interval == 0:
                    print(f"\tStep {i} / {len(dataloader)}.\tLoss: {loss.item():.4f}")

                    self.log(model.state_dict(), loss=loss.item())

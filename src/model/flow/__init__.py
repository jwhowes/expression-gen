from src.config import Config
from .dit import DiT


class FlowMatchModelConfig(Config):
    latent_channels: int
    num_classes: int

    d_model: int
    d_t: int
    n_layers: int
    n_heads: int

    def get_model(self) -> DiT:
        return DiT(
            d_in=self.latent_channels,
            num_classes=self.num_classes,
            d_model=self.d_model,
            d_t=self.d_t,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
        )

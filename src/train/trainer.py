import os
import yaml
import torch

from datetime import datetime
from pydantic import Field
from abc import ABC, abstractmethod
from typing import Tuple, Type

from src.config import Config


class OptimizerConfig(Config):
    lr: float
    weight_decay: float
    clip_grad: float = Field(default=3.0)


EXPERIMENTS_ROOT = "experiments"


class Trainer(ABC):
    config_type: Type[Config]
    metrics: Tuple[str]

    def __init__(self, config_path: str):
        exp_name = os.path.splitext(os.path.basename(config_path))[0]
        self.config = self.config_type.from_yaml(config_path)

        self.exp_dir = os.path.join(EXPERIMENTS_ROOT, exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")

        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        with open(os.path.join(self.exp_dir, "config.yaml"), "w+") as f:
            yaml.dump(self.config.model_dump(), f)

        self.ckpt = 1

        self.log_path = os.path.join(self.exp_dir, "log.csv")
        with open(self.log_path, "w+") as f:
            f.write(f"ckpt,{','.join([m for m in self.metrics])},timestamp\n")

    def log(self, state_dict: dict, **metrics: float):
        with open(self.log_path, "a") as f:
            f.write(
                f"{self.ckpt},{','.join([f'{metrics[m]:.4f}' for m in self.metrics])},{datetime.now()}\n"
            )

        torch.save(
            state_dict, os.path.join(self.ckpt_dir, f"checkpoint_{self.ckpt:03}.pth")
        )

        self.ckpt += 1

    @abstractmethod
    def train(self): ...

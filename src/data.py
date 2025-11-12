import os

from PIL import Image
from pydantic import Field
from torch import Tensor
from torch.utils.data import Dataset
from typing import Literal, Tuple
from torchvision import transforms

from .config import Config

DATA_DIR = "data"


class ExpressionDataset(Dataset):
    def __init__(
        self, image_size: int = 48, split: Literal["train", "validation"] = "train"
    ):
        data_root = os.path.join(DATA_DIR, split)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(image_size),
                transforms.Normalize(mean=(0.5060,), std=(0.2506,)),
            ]
        )

        self.files = []
        self.labels = []

        self.label_to_id = {}

        for i, expression in enumerate(os.listdir(data_root)):
            for file in os.listdir(os.path.join(data_root, expression)):
                self.files.append(os.path.join(data_root, expression, file))
                self.labels.append(i)

                self.label_to_id[i] = expression

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        return self.transform(Image.open(self.files[idx])), self.labels[idx]


class ExpressionDatasetConfig(Config):
    batch_size: int

    image_size: int = Field(default=48)
    split: Literal["train", "validation"] = Field(default="train")

    def get_dataset(self) -> ExpressionDataset:
        return ExpressionDataset(image_size=self.image_size, split=self.split)

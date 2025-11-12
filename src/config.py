import yaml

from typing import Self
from pydantic import BaseModel


class Config(BaseModel):
    @classmethod
    def from_yaml(cls, yaml_path: str) -> Self:
        with open(yaml_path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        return cls.model_validate(data)

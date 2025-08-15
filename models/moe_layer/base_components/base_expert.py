from models.base_model import BaseModel

from abc import abstractmethod


class BaseExpert(BaseModel):
    def __init__(self, in_channels, out_channels, name=None, num_layers=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name
        self.num_layers = num_layers

    @abstractmethod
    def forward(self, x):
        pass
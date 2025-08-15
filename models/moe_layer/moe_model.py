import torch
import torchvision
import torch.nn as nn

import time
import os
import copy

from models.base_model import BaseModel
from models.moe_layer.moe_layer import MoE_Layer

class MoE_Model(BaseModel):
    def __init__(self, num_classes, num_experts, name='mnist_moe', device='cuda:0'):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.device=device

        self.moe_layer1 = MoE_Layer(in_channels=1, out_channels=64, num_experts=self.num_experts, top_k=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(26 * 26 * 64, 128)
        self.fc2 = nn.Linear(128, self.num_classes)


    def forward(self, x):
        moe_output = self.moe_layer1(x)
        x, aux_loss1 = moe_output['output'], moe_output['aux_loss']
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return {'output': x, 'aux_loss': aux_loss1}
        
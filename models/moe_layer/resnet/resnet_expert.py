import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet
from models.moe_layer.base_components.base_expert import BaseExpert

class ResNetExpert(BaseExpert):
    def __init__(self, in_channels, out_channels, name=None, num_blocks=1, stride=1):
        super().__init__(in_channels, out_channels, name)

        # Build expert model
        self.model = self._build_model(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_blocks=num_blocks,
            stride=stride
        )

        #self.model = self._make_layer(block=BasicBlock, planes=64, blocks=2)


    def forward(self, x, output_only=True):
        out = self.model(x)
        if output_only:
            return out
        else:
            return {'output': out}


    def _build_model(self, in_channels, out_channels, num_blocks, stride=1):
        # Initialize sequential model
        layers = nn.ModuleList()

        # Add first ResNet block
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        block1 = BasicBlock(in_channels, out_channels, stride=stride, downsample=downsample)
        layers.append(block1)

        # Create additional ResNet blocks according to num_layers
        for i in range(1, num_blocks):
            block = BasicBlock(out_channels, out_channels, stride=1)
            layers.append(block)

        return nn.Sequential(*layers)

import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet
from models.moe_layer.base_components.base_expert import BaseExpert
from models.base_model import BaseModel
from torchvision.models.resnet import resnet101, resnet50
from copy import copy
from torchsummary import summary


class NarrowResNet18Expert(BaseModel):
    def __init__(self, layer_position, name=None):
        super().__init__(name)
        self.layer_position = layer_position
        
        if layer_position == 0:
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )

        elif layer_position == 1:
            self.model = self._make_layer(1, 2)
        elif layer_position == 2:
            self.model = self._make_layer(2, 2)
        elif layer_position == 3:
            self.model = self._make_layer(3, 2)
        elif layer_position == 4:
            self.model = self._make_layer(4, 2)
        else:
            raise RuntimeError('Invalid layer number: ', layer_position)

    def forward(self, x, output_only=True):
        out = self.model.forward(x)
        if output_only:
            return out
        else:
            return {'output': out}
    
    def _make_layer(self, layer_position, num_blocks):
        # planes: num channels in intermediate layers
        # inplanes: num channels in input layer
        # outpkanes: num channels in last layer
        if layer_position == 1:
            planes = 25
            inplanes = 64
            outplanes = 64
            stride = 1
        elif layer_position == 2:
            planes = 50
            inplanes = 64
            outplanes = 128
            stride = 2
        elif layer_position == 3:
            planes = 90
            inplanes = 128
            outplanes = 256
            stride = 2
        elif layer_position == 4:
            planes = 180
            inplanes = 256
            outplanes = 512
            stride = 2
        else:
            raise RuntimeError('Invalid layer number: ', layer_position)

        norm_layer = nn.BatchNorm2d
        block = BasicBlock
        downsample = None
        previous_dilation = 1
        dilitation = 1
        base_width = 64
        groups = 1


        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, groups,
                            base_width, previous_dilation, norm_layer))

        for _ in range(1, num_blocks - 1):
            layers.append(block(planes, planes, groups=groups))
        
        downsample = None
        if planes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(planes, outplanes, kernel_size=1, stride=1, bias=False),
                norm_layer(outplanes),
            )

        # Use BasicBlock without ReLU as last block
        # block = BasicBlockWithoutReLU
        layers.append(block(planes, outplanes, stride=1, downsample=downsample, groups=groups,))

        return nn.Sequential(*layers)


class BasicBlockWithoutReLU(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # Remove ReLU

        return out

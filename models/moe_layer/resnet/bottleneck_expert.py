import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet
from models.moe_layer.base_components.base_expert import BaseExpert
from models.base_model import BaseModel
from torchvision.models.resnet import resnet101, resnet50
from copy import copy
from torchsummary import summary

class BottleneckExpert(BaseExpert):
    def __init__(self, in_channels, out_channels, name=None):
        super().__init__(in_channels, out_channels, name)

        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        )


        # Build expert model
        self.model = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64, dilation=4),
            Bottleneck(256, 64, dilation=4)
        )


    def forward(self, x, output_only=True):
        out = self.model(x)
        if output_only:
            return out
        else:
            return {'output': out}

class ResNet101Expert(BaseModel):
    def __init__(self, layer_position, name=None):
        super().__init__(name)
        self.layer_position = layer_position
        resnet = resnet101(pretrained=False)
        if layer_position == 0:
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        elif layer_position == 1:
            self.model = resnet.layer1
        elif layer_position == 2:
            self.model = resnet.layer2
        elif layer_position == 3:
            self.model = resnet.layer3
        elif layer_position == 4:
            self.model = resnet.layer4
        else:
            raise RuntimeError('Invalid layer number: ', layer_position)

    def forward(self, x, output_only=True):
        out = self.model.forward(x)
        if output_only:
            return out
        else:
            return {'output': out}


class ResNet50Expert(BaseModel):
    def __init__(self, layer_position, name=None):
        super().__init__(name)
        self.layer_position = layer_position
        resnet = resnet50(pretrained=False)
        
        if layer_position == 0:
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        elif layer_position == 1:
            self.model = resnet.layer1
        elif layer_position == 2:
            self.model = resnet.layer2
        elif layer_position == 3:
            self.model = resnet.layer3
        elif layer_position == 4:
            self.model = resnet.layer4
        else:
            raise RuntimeError('Invalid layer number: ', layer_position)

    def forward(self, x, output_only=True):
        out = self.model.forward(x)
        if output_only:
            return out
        else:
            return {'output': out}

class NarrowResNet50Expert(BaseModel):
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
            self.model = self._make_layer(1, 3)
        elif layer_position == 2:
            self.model = self._make_layer(2, 4)
        elif layer_position == 3:
            self.model = self._make_layer(3, 6)
        elif layer_position == 4:
            self.model = self._make_layer(4, 3)
        else:
            raise RuntimeError('Invalid layer number: ', layer_position)

    def forward(self, x, output_only=True):
        out = self.model.forward(x)
        if output_only:
            return out
        else:
            return {'output': out}
    
    def _make_layer(self, layer_position, blocks):
        if layer_position == 1:
            planes = 24
            inplanes = 64
            outplanes = 256
            stride = 1
        elif layer_position == 2:
            planes = 58
            inplanes = 256
            outplanes = 512
            stride = 2
        elif layer_position == 3:
            planes = 140
            inplanes = 512
            outplanes = 1024
            stride = 2
        elif layer_position == 4:
            planes = 200
            inplanes = 1024
            outplanes = 2048
            stride = 2
        else:
            raise RuntimeError('Invalid layer number: ', layer_position)

        norm_layer = nn.BatchNorm2d
        block = Bottleneck
        downsample = None
        previous_dilation = 1
        dilitation = 1
        base_width = 64
        groups = 1

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, groups,
                            base_width, previous_dilation, norm_layer))

        for _ in range(1, blocks - 1):
            layers.append(block(planes * block.expansion, planes, groups=groups))
        

        downsample = nn.Sequential(
                nn.Conv2d(planes * block.expansion, outplanes, kernel_size=1, stride=1, bias=False),
                norm_layer(outplanes),
            )

        layers.append(block(planes * block.expansion, outplanes // 4, stride=1, downsample=downsample, groups=groups,))

        return nn.Sequential(*layers)

class ShallowResNet50Expert(BaseModel):
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
            self.model = self._make_layer(3, 3)
        elif layer_position == 4:
            self.model = self._make_layer(4, 1)
        else:
            raise RuntimeError('Invalid layer number: ', layer_position)

    def forward(self, x, output_only=True):
        out = self.model.forward(x)
        if output_only:
            return out
        else:
            return {'output': out}
    
    def _make_layer(self, layer_position, blocks):
        if layer_position == 1:
            planes = 64
            inplanes = 64
            stride = 1
        elif layer_position == 2:
            planes = 128
            inplanes = 256
            stride = 2
        elif layer_position == 3:
            planes = 256
            inplanes = 512
            stride = 2
        elif layer_position == 4:
            planes = 512
            inplanes = 1024
            stride = 2
        else:
            raise RuntimeError('Invalid layer number: ', layer_position)

        norm_layer = nn.BatchNorm2d
        block = Bottleneck
        downsample = None
        previous_dilation = 1
        dilitation = 1
        base_width = 64
        groups = 1

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, groups,
                            base_width, previous_dilation, norm_layer))
       
        for _ in range(1, blocks):
            layers.append(block(planes *  block.expansion, planes, groups=groups))

        return nn.Sequential(*layers)

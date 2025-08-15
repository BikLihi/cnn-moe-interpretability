import torch
import torch.nn as nn

from torchvision.models.resnet import resnet101

from models.base_model import BaseModel

from utils.cifar100_utils import CIFAR100_LABELS

from ptflops import get_model_complexity_info


class Resnet101(BaseModel):

    def __init__(self, classes, in_channels=3, name='ResNet101', *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.classes = classes
        self.name = name
        self.in_channels = in_channels
        self.num_classes = len(self.classes)
        self.custom_config['name'] = name

        # Load ResNet18 architecture
        resnet = resnet101(pretrained=False)

        # Adjust first conv layer and removing maxpool layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu

        # Adding remaining ResNet18 layers
        self.layers = nn.ModuleList()
        self.layers.append(resnet.layer1)
        self.layers.append(resnet.layer2)
        self.layers.append(resnet.layer3)
        self.layers.append(resnet.layer4)
        
        # Adjust number of output neurons in fc layer
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)


    def forward(self, x, output_only=True):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layers[0](out)
        out = self.layers[1](out)
        out = self.layers[2](out)
        out = self.layers[3](out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        if output_only:
            return out
        else:
            return {'output': out}

from models.moe_layer.base_components.base_moe_layer import BaseMoELayer
from models.moe_layer.resnet.bottleneck_expert import BottleneckExpert
import torch.nn as nn

class MoeBlockLayer(BaseMoELayer):
    def __init__(self, num_experts, layer_position, top_k, gating_network, resnet_expert):
        experts = nn.ModuleList([resnet_expert(layer_position) for i in range(num_experts)])
        super().__init__(
            gate=gating_network,
            num_experts=num_experts,
            top_k=top_k,
            experts=experts)

        self.layer_position = layer_position

        if self.num_experts == 1:
            self.gate.set_parameter_requires_grad(False)


    def forward(self, x, output_only=True):
        return super().forward(x, output_only)


class ResidualMoeBlockLayer(MoeBlockLayer):
    def __init__(self, num_experts, layer_position, top_k, gating_network, resnet_expert):
        super().__init__(num_experts, layer_position, top_k, gating_network, resnet_expert)

        channels = [64, 64, 128, 256, 512]
        stride = [1, 2, 2, 2]

        if self.layer_position == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                    nn.Conv2d(channels[self.layer_position - 1], channels[self.layer_position], kernel_size=1, stride=stride[self.layer_position - 1], bias=False),
                    nn.BatchNorm2d(channels[self.layer_position]),
                )
    


    def forward(self, x, output_only=True):
        moe_output = super().forward(x, output_only)
        identity = self.downsample(x)
        if type(moe_output) is dict:
            moe_output['output'] += identity
        else:
            moe_output += identity
        return moe_output
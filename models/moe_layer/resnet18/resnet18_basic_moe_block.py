from torchvision.models.resnet import BasicBlock
from models.moe_layer.resnet.moe_conv_layer import MoEConvLayer
from models.moe_layer.soft_gating_networks import SimpleGate
from models.base_model import BaseModel
import torch.nn as nn
import torch

class BasicMoEBlock(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, num_Experts=4, top_k=2, loss_fkt='importance', w_aux_loss=0.1, ):
        super().__init__(inplanes, planes, stride, downsample,
                         groups=1, base_width=64, dilation=1, norm_layer=None)
        
        self.conv1 = MoEConvLayer(num_Experts, top_k, SimpleGate, loss_fkt, w_aux_loss, self.conv1.in_channels, self.conv1.out_channels,
                                  self.conv1.kernel_size, self.conv1.stride, self.conv1.padding, self.conv1.bias)

        self.conv2 = MoEConvLayer(num_Experts, top_k, SimpleGate, loss_fkt, w_aux_loss, self.conv2.in_channels, self.conv2.out_channels,
                                  self.conv2.kernel_size, self.conv2.stride, self.conv2.padding, self.conv2.bias)

    def forward(self, x, output_only=True):
        out = super().forward(x)
        if output_only:
            return out
        aux_loss = self.conv1.current_aux_loss + self.conv2.current_aux_loss
        examples_per_expert = [self.conv1.current_examples_per_expert, self.conv2.current_examples_per_expert]
        expert_importance = [self.conv1.current_expert_importance, self.conv2.current_expert_importance]
        weights = [self.conv1.current_weights, self.conv2.current_weights]
        return {'output': out, 'aux_loss': aux_loss, 'examples_per_expert': examples_per_expert, 'expert_importance': expert_importance, 'weights': weights}


class ResnetMultiMoEBlock(BaseModel):
    def __init__(self, layer_position, num_experts=4, top_k=2, loss_fkt='importance', w_aux_loss=0.1):
        super().__init__('ResnetMultiMoEBlock')
        self.layers = self._make_layer(layer_position, num_experts, top_k, loss_fkt, w_aux_loss)
    
    def forward(self, x, output_only=True):
        out = x
        aux_loss = torch.tensor(0.0, device=self.device)
        examples_per_expert = []
        expert_importance = []
        weights = []
        if output_only:
            for layer in self.layers:
                out = layer(out, output_only=True)
            return out
        else:
            for layer in self.layers:
                output = layer(out, output_only=False)
                aux_loss = aux_loss + output['aux_loss']
                examples_per_expert.append(output['examples_per_expert'])
                expert_importance.append(output['expert_importance'])
                weights.append(output['weights'])
                out = output['output']
        return {'output': out, 'aux_loss': aux_loss, 'examples_per_expert': examples_per_expert, 'expert_importance': expert_importance}

    def _make_layer(self, layer_position, num_experts=4, top_k=2, loss_fkt='importance', w_aux_loss=0.1):
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
        block = BasicMoEBlock
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
        layers.append(block(inplanes, planes, stride, downsample, num_experts, top_k, loss_fkt, w_aux_loss))

        downsample = None
        if planes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(planes, outplanes, kernel_size=1, stride=1, bias=False),
                norm_layer(outplanes),
            )


        layers.append(block(planes, outplanes, 1, downsample, num_experts, top_k, loss_fkt, w_aux_loss))

        return nn.Sequential(*layers)

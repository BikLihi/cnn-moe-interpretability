import torch.nn as nn

from models.base_model import BaseModel
from models.moe_layer.resnet.resnet_moe_layer import ResNetMoeLayer


class ShallowResNetMoE(BaseModel):
    def __init__(self, num_experts=None, top_k=None, use_noise=True, num_blocks=1, gating_network=None, moe_layer=None, name='ShallowResNetMoE', loss_fkt=None, w_aux_loss=None, constr_threshold=None):
        super().__init__(name=name)

        self.custom_config['num_experts'] = num_experts
        self.custom_config['num_blocks'] = num_blocks
        self.custom_config['top_k'] = top_k
        self.custom_config['gating_network'] = gating_network
        self.custom_config['loss_fkt'] = loss_fkt
        self.custom_config['w_aux_loss'] = w_aux_loss
        self.custom_config['constr_threshold'] = constr_threshold

        # Generic input block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # ResNet MoE Layer
        if moe_layer is None:
            self.moe = ResNetMoeLayer(
                in_channels=64,
                out_channels=128,
                num_experts=num_experts, 
                top_k=top_k,
                num_blocks=num_blocks,
                use_noise=use_noise,
                gating_network=gating_network,
                loss_fkt=loss_fkt,
                w_aux_loss=w_aux_loss,
                constr_threshold=constr_threshold
            )
        else:
            self.moe = moe_layer

        # Generic output block
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128, out_features=100, bias=True)

    def forward(self, x, output_only=True):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if output_only:
            out = self.moe(out)
            out = self.avgpool(out)
            out = self.flatten(out)
            out = self.fc(out)
            return out

        moe_output = self.moe(out, output_only=False)
        out = moe_output['output']
        aux_loss = moe_output['aux_loss']
        examples_per_expert = moe_output['examples_per_expert']
        expert_importance = moe_output['expert_importance']

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)

        return {'output': out, 'aux_loss': aux_loss, 'examples_per_expert': examples_per_expert, 'expert_importance': expert_importance}

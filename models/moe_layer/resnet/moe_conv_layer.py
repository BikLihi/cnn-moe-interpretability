from models.moe_layer.base_components.base_moe_layer import BaseMoELayer
from models.moe_layer.hard_gating_networks import RelativeImportanceGate, AbsoluteImportanceGate
import torch.nn as nn


class MoEConvLayer(BaseMoELayer):
    def __init__(self, num_experts, top_k, gating_network, loss_fkt, w_aux_loss, constr_threshold, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
        experts = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, bias=bias) for i in range(num_experts)])
        
        if type(gating_network) == RelativeImportanceGate or type(gating_network) == AbsoluteImportanceGate:
            gate = gating_network(in_channels, num_experts, top_k, constr_threshold)
        else:
            gate = gating_network(in_channels, num_experts, top_k, loss_fkt, w_aux_loss)
        self.current_aux_loss = None
        self.current_examples_per_expert = None,
        self.current_expert_importance = None
        self.current_weights = None

        super().__init__(
            gate=gate,
            num_experts=num_experts,
            top_k=top_k,
            experts=experts)

    def forward(self, x, output_only=True):
        out = super().forward(x, output_only=False)
        self.current_aux_loss = out['aux_loss']
        self.current_examples_per_expert = out['examples_per_expert'],
        self.current_expert_importance = out['expert_importance']
        self.current_weights = out['weights']
        return out['output']
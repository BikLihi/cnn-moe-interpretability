import torch
from models.moe_layer.base_components.base_gating_network import BaseGatingNetwork

class EqualWeightGatingNetwork(BaseGatingNetwork):
    def __init__(self, in_channels, num_experts=8, name='EqualWeightGatingNetwork', **kwargs):
        super().__init__(
            in_channels=in_channels,
            num_experts=num_experts,
            top_k=num_experts,
            use_noise=False,
            name=name,
            loss_fkt=None,
            w_aux_loss=None,
        )

    def forward(self, x, output_only=True):
        out = torch.full((x.shape[0], self.num_experts), fill_value=1.0/self.num_experts)
        if output_only:
            return out
        else:
            return {'output': out}

    def compute_gating(self, x):
        return self(x)

class SingleWeightingGatingNetwork(BaseGatingNetwork):
    def __init__(self, in_channels, expert_index, num_experts, name='SingleWeightGatingNetwork', **kwargs):
        super().__init__(
            in_channels=in_channels,
            num_experts=num_experts,
            top_k=num_experts,
            use_noise=False,
            name=name,
            loss_fkt=None,
            w_aux_loss=None,
        )
        self.expert_index = expert_index

    def forward(self, x, output_only=True):
        out = torch.full((x.shape[0], self.num_experts), fill_value=0.0)
        out = out.to('cuda:0')
        out[:, self.expert_index] = 1.0
        if output_only:
            return out
        else:
            return {'output': out}

    def compute_gating(self, x):
        return self(x)
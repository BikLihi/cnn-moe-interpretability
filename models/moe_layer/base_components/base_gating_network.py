from models.base_model import BaseModel

from abc import abstractmethod
import torch.nn as nn
import torch

from losses.importance_loss import importance, importance_loss
from losses.kullback_leibler_divergence import kl_divergence

class BaseGatingNetwork(BaseModel):
    def __init__(self, in_channels, num_experts, top_k, use_noise=False, name='Gate', loss_fkt=None, w_aux_loss=None, constr_threshold=None, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.name = name
        self.use_noise = use_noise
        self.loss_fkt = loss_fkt
        self.w_aux_loss = w_aux_loss
        self.constr_threshold = constr_threshold

        self.avgpool_1x1 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.w_noise = nn.Linear(in_channels, self.num_experts, bias=False)
        self.softplus = nn.Softplus()


    @abstractmethod
    def forward(self, x):      
        pass


    def compute_loss(self, x):
        if self.w_aux_loss is None or self.loss_fkt is None:
            return 0.0
        elif self.loss_fkt == 'importance':
            return importance_loss(x) * self.w_aux_loss
        elif self.loss_fkt == 'kl_divergence':
            return kl_divergence(x, self.num_experts) * self.w_aux_loss
        else:
            raise NameError('Loss function not found')

    
    def compute_gating(self, x):
        gate_logits = self(x)

        if self.num_experts == 1:
            return gate_logits, x.shape[0]
            
        # Apply noise during training time if parameter is set
        if self.use_noise and self.training:
            pool_out = self.avgpool_1x1(x)
            flatten_out = self.flatten(pool_out)
            std = self.softplus(self.w_noise(flatten_out) + 1e-2)
            noise = torch.randn_like(gate_logits, device=self.device, requires_grad=True) * std
            gate_logits += noise
        
        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=1)
        top_k_weights = nn.functional.softmax(top_k_logits, dim=1)

        weight_zeros = torch.zeros_like(gate_logits, device=self.device, requires_grad=True)
        weights = weight_zeros.scatter(1, top_k_indices, top_k_weights)
        return weights

from models.moe_layer.base_components.base_gating_network import BaseGatingNetwork
import torch.nn as nn
import torch
import numpy as np
import math
from losses.importance_loss import importance, importance_loss
from losses.kullback_leibler_divergence import kl_divergence


class ConvGate(BaseGatingNetwork):
    def __init__(self, in_channels, num_experts=8, top_k=2, use_noise=True, name='ConvGate', loss_fkt=None, w_aux_loss=None):
        # Set attributes
        super().__init__(
            in_channels=in_channels,
            num_experts=num_experts,
            top_k=top_k,
            use_noise=use_noise,
            name=name,
            loss_fkt=loss_fkt,
            w_aux_loss=w_aux_loss)
        # Add layer for gate computation
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1)
        self.fc = nn.Linear(256, self.num_experts)


    def forward(self, x, output_only=True):
        if self.num_experts == 1:
            return torch.ones((x.shape[0], 1))

        out = self.conv1(x)
        out = self.avgpool_1x1(out)
        out = self.flatten(out)
        out = self.fc(out)

        if output_only:
            return out
        else:
            return {'output': out}


class ConvGateVarianceLoss(BaseGatingNetwork):
    def __init__(self, in_channels, num_experts=8, top_k=2, use_noise=True, name='ConvGate', loss_fkt=None, w_aux_loss=None, w_variance_loss=0.0):
        # Set attributes
        super().__init__(
            in_channels=in_channels,
            num_experts=num_experts,
            top_k=top_k,
            use_noise=use_noise,
            name=name,
            loss_fkt=loss_fkt,
            w_aux_loss=w_aux_loss)
        self.w_variance_loss = w_variance_loss
        # Add layer for gate computation
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1)
        self.fc = nn.Linear(256, self.num_experts)


    def forward(self, x, output_only=True):
        if self.num_experts == 1:
            return torch.ones((x.shape[0], 1))

        out = self.conv1(x)
        out = self.avgpool_1x1(out)
        out = self.flatten(out)
        out = self.fc(out)

        if output_only:
            return out
        else:
            return {'output': out}

    def compute_loss(self, x):
        mean_weightings = x.mean(dim=0)
        variances = torch.square(x - mean_weightings).sum(dim=0) / (self.num_experts - 1)
        var_loss = -variances.mean(dim=0) * self.w_variance_loss

        if self.loss_fkt == 'importance':
            aux_loss = importance_loss(x) * self.w_aux_loss
        elif self.loss_fkt == 'kl_divergence':
            aux_loss = kl_divergence(x, self.num_experts) * self.w_aux_loss
        else:
            raise NameError('Loss function not found')
        return var_loss + aux_loss



class SimpleGate(BaseGatingNetwork):
    def __init__(self, in_channels, num_experts=8, top_k=2, use_noise=True, name='SimpleGate', loss_fkt='importance', w_aux_loss=0.1, **kwargs):
        super().__init__(
            in_channels=in_channels,
            num_experts=num_experts,
            top_k=top_k,
            use_noise=use_noise,
            name=name,
            loss_fkt=loss_fkt,
            w_aux_loss=w_aux_loss)
        # Add layer for gate computation
        self.fc = nn.Linear(in_channels, self.num_experts)

    def forward(self, x, output_only=True):
        if self.num_experts == 1:
            return torch.ones((x.shape[0], 1))

        out = self.avgpool_1x1(x)
        out = self.flatten(out)
        out = self.fc(out)

        if output_only:
            return out
        else:
            return {'output': out}


class SimpleGateVarianceLoss(BaseGatingNetwork):
    def __init__(self, in_channels, w_variance_loss, num_experts=8, top_k=2, use_noise=True, name='SimpleGate', loss_fkt='importance', w_aux_loss=0.1, **kwargs):
        super().__init__(
            in_channels=in_channels,
            num_experts=num_experts,
            top_k=top_k,
            use_noise=use_noise,
            name=name,
            loss_fkt=loss_fkt,
            w_aux_loss=w_aux_loss)
        # Add layer for gate computation
        self.w_variance_loss = w_variance_loss
        self.fc = nn.Linear(in_channels, self.num_experts)

    def forward(self, x, output_only=True):
        if self.num_experts == 1:
            return torch.ones((x.shape[0], 1))

        out = self.avgpool_1x1(x)
        out = self.flatten(out)
        out = self.fc(out)

        if output_only:
            return out
        else:
            return {'output': out}

    def compute_loss(self, x):
        mean_weightings = x.mean(dim=0)
        variances = torch.square(x - mean_weightings).sum(dim=0)
        var_loss = -(variances / x.shape[0]).sum(dim=0) * self.w_variance_loss

        if self.loss_fkt == 'importance':
            aux_loss = importance_loss(x) * self.w_aux_loss
        elif self.loss_fkt == 'kl_divergence':
            aux_loss = kl_divergence(x, self.num_experts) * self.w_aux_loss
        else:
            raise NameError('Loss function not found')
        return var_loss + aux_loss


class FCGate(BaseGatingNetwork):
    def __init__(self, in_channels, num_experts=8, top_k=2, use_noise=True, name='SimpleGate', loss_fkt='importance', w_aux_loss=0.1, **kwargs):
        super().__init__(
            in_channels=in_channels,
            num_experts=num_experts,
            top_k=top_k,
            use_noise=use_noise,
            name=name,
            loss_fkt=loss_fkt,
            w_aux_loss=w_aux_loss)
        # Add layer for gate computation
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_channels, self.num_experts)

    def forward(self, x, output_only=True):
        if self.num_experts == 1:
            return torch.ones((x.shape[0], 1))

        out = self.conv1(x)
        out = self.relu(x)
        out = self.avgpool_1x1(out)
        out = self.flatten(out)
        out = self.fc(out)

        if output_only:
            return out
        else:
            return {'output': out}

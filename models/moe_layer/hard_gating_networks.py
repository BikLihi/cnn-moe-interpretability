from models.moe_layer.base_components.base_gating_network import BaseGatingNetwork
import torch.nn as nn
import torch
import numpy as np
import math
from losses.importance_loss import importance, importance_loss


class RelativeImportanceGate(BaseGatingNetwork):
    def __init__(self, in_channels, num_experts=8, top_k=2, use_noise=True, name='RelativeImportanceGate', constr_threshold=0.1, **kwargs):
        # Set attributes
        super().__init__(
            in_channels=in_channels,
            num_experts=num_experts,
            top_k=top_k,
            use_noise=use_noise,
            name=name,
            constr_threshold=constr_threshold)
        # Add layer for gate computation
        self.fc = nn.Linear(in_channels, self.num_experts)
        # Attributres for constraint computation
        self.latest_importance = torch.zeros(self.num_experts, device=self.device, requires_grad=False).detach()
        self.running_rel_importance = torch.zeros(self.num_experts, device=self.device, requires_grad=False).detach()
        self.step_counter = 0
        self.constr_threshold = constr_threshold


    def forward(self, x, output_only=True):
        if self.num_experts == 1:
            return torch.ones((x.shape[0], 1), device=self.device)

        out = self.avgpool_1x1(x)
        out = self.flatten(out)
        out = self.fc(out)

        if output_only:
            return out
        else:
            return {'output': out}


    def compute_gating(self, x):
        gate_logits = self(x)

        if self.num_experts == 1:
            return gate_logits, x.shape[0]

        # Check hard constraint on expert importance during training time
        if self.training:
            for i, g in enumerate(self.running_rel_importance):
                if g > self.constr_threshold:
                    gate_logits[:, i] = -float("inf")

        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=1)
        top_k_weights = nn.functional.softmax(top_k_logits, dim=1)

        weight_zeros = torch.zeros_like(gate_logits, device=self.device, requires_grad=True)
        weights = weight_zeros.scatter(1, top_k_indices, top_k_weights)

        # Update running importance and avg importance during training time
        expert_importance = importance(weights)
        expert_importance = expert_importance.detach()
        if self.training:
            avg_importance = x.shape[0] / self.num_experts
            self.running_rel_importance = (self.running_rel_importance + (expert_importance - avg_importance) / avg_importance).data

        return weights


    def compute_loss(self, x):
        return 0.0


class RelativeImportanceGatConv(BaseGatingNetwork):
    def __init__(self, in_channels, num_experts=8, top_k=2, use_noise=True, name='RelativeImportanceGate', constr_threshold=0.1, **kwargs):
        # Set attributes
        super().__init__(
            in_channels=in_channels,
            num_experts=num_experts,
            top_k=top_k,
            use_noise=use_noise,
            name=name,
            constr_threshold=constr_threshold)
        # Add layer for gate computation
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1)
        self.fc = nn.Linear(256, self.num_experts)
        # Attributres for constraint computation
        self.latest_importance = torch.zeros(self.num_experts, device=self.device, requires_grad=False).detach()
        self.running_rel_importance = torch.zeros(self.num_experts, device=self.device, requires_grad=False).detach()
        self.step_counter = 0
        self.constr_threshold = constr_threshold


    def forward(self, x, output_only=True):
        if self.num_experts == 1:
            return torch.ones((x.shape[0], 1), device=self.device)

        out = self.conv1(x)
        out = self.avgpool_1x1(out)
        out = self.flatten(out)
        out = self.fc(out)

        if output_only:
            return out
        else:
            return {'output': out}



class AbsoluteImportanceGate(BaseGatingNetwork):
    def __init__(self, in_channels, num_experts=8, top_k=2, use_noise=True, name='AbsoluteImportanceGate', constr_threshold=0.1, **kwargs):
        # Set attributes
        super().__init__(
            in_channels=in_channels,
            num_experts=num_experts,
            top_k=top_k,
            use_noise=use_noise,
            name=name,
            constr_threshold=constr_threshold)
        # Add layer for gate computation
        self.fc = nn.Linear(in_channels, self.num_experts)
        # Attributres for constraint computation
        self.running_rel_importance = torch.zeros(self.num_experts, device=self.device)
        self.step_counter = 0


    def forward(self, x, output_only=True):
        if self.num_experts == 1:
            return torch.ones((x.shape[0], 1), device=self.device)

        out = self.avgpool_1x1(x)
        out = self.flatten(out)
        out = self.fc(out)

        if output_only:
            return out
        else:
            return {'output': out}


    def compute_gating(self, x):
        gate_logits = self(x)

        if self.num_experts == 1:
            return gate_logits, x.shape[0]

        # Check hard constraint on expert importance during training time
        if self.training:
            avg_imp = 1.0 / self.num_experts
            for i, imp in enumerate(self.running_rel_importance):
                if imp - avg_imp > self.constr_threshold:
                    gate_logits[:, i] = -float("inf")

        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=1)
        top_k_weights = nn.functional.softmax(top_k_logits, dim=1)

        weight_zeros = torch.zeros_like(gate_logits, device=self.device, requires_grad=True)
        weights = weight_zeros.scatter(1, top_k_indices, top_k_weights)
        # Update running importance and avg importance during training time
        if self.training:
            perc_expert_imp = importance(weights) / x.shape[0]
            self.running_rel_importance = ((self.running_rel_importance * self.step_counter) + perc_expert_imp) / (self.step_counter + 1)
            self.step_counter += 1
        return weights


    def compute_loss(self, x):
        return 0.0


class FCRelativeImportanceGate(BaseGatingNetwork):
    def __init__(self, in_channels, num_experts=8, top_k=2, use_noise=True, name='RelativeImportanceGate', constr_threshold=0.1, **kwargs):
        # Set attributes
        super().__init__(
            in_channels, 
            num_experts, 
            top_k, 
            use_noise, 
            name, 
            constr_threshold)

        # Add layer for gate computation
        self.fc = nn.Linear(in_channels, self.num_experts)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Attributres for constraint computation
        self.running_rel_importance2 = torch.zeros(self.num_experts, device='cpu:0', requires_grad=False).detach()        
        self.running_rel_importance = np.zeros(self.num_experts)

        self.constr_threshold = constr_threshold


        # Add layer for gate computation

    def forward(self, x, output_only=True):
        if self.num_experts == 1:
            return torch.ones((x.shape[0], 1), device=self.device)

        out = self.conv1(x)
        out = self.relu(x)
        out = self.avgpool_1x1(out)
        out = self.flatten(out)
        out = self.fc(out)

        if output_only:
            return out
        else:
            return {'output': out}

    def compute_gating(self, x):
        gate_logits = self(x)

        if self.num_experts == 1:
            return gate_logits, x.shape[0]

        # Check hard constraint on expert importance during training time
        if self.training:
            for i, g in enumerate(self.running_rel_importance):
                if g > self.constr_threshold:
                    gate_logits[:, i] = -float("inf")

        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=1)
        top_k_weights = nn.functional.softmax(top_k_logits, dim=1)

        weight_zeros = torch.zeros_like(gate_logits, device=self.device, requires_grad=True)
        weights = weight_zeros.scatter(1, top_k_indices, top_k_weights)

        # Update running importance and avg importance during training time
        expert_importance = importance(weights).detach().cpu().numpy()
        if self.training:
            avg_importance = x.shape[0] / self.num_experts
            self.running_rel_importance = (self.running_rel_importance + (expert_importance - avg_importance) / avg_importance)
        return weights

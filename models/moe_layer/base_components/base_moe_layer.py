import torch
import torch.nn as nn

from models.base_model import BaseModel

from losses.importance_loss import importance_loss, importance


class BaseMoELayer(BaseModel):

    def __init__(self, gate, num_experts=8, top_k=2, experts=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = experts
        self.gate = gate

    def forward(self, x, output_only=True):
        for expert in self.experts:
            expert.to(self.device)

        if self.num_experts == 1:
            out = self.experts[0](x)
            if output_only:
                return out
            return {'output': out,
                    'aux_loss': 0.0,
                    'examples_per_expert': torch.Tensor([x.shape[0]]),
                    'expert_importance': torch.tensor([1.0]),
                    'weights': torch.ones(x.shape[0])}

        weights = self.gate.compute_gating(x)
        examples_per_expert = (weights > 0).sum(dim=0)
        expert_importance = importance(weights)

        aux_loss = self.gate.compute_loss(weights)
        mask = weights > 0
        results = []
        for i in range(self.num_experts):
            # select mask according to computed gates (conditional computing)
            mask_expert = mask[:, i]
            # apply mask to inputs
            expert_input = x[mask_expert]
            # compute outputs for selected examples
            expert_output = self.experts[i](expert_input).to(self.device)
            # calculate output shape
            output_shape = list(expert_output.shape)
            output_shape[0] = x.size()[0]
            # store expert results in matrix
            expert_result = torch.zeros(output_shape, device=self.device)
            expert_result[mask_expert] = expert_output
            # weight expert's results
            expert_weight = weights[:, i].reshape(
                expert_result.shape[0], 1, 1, 1).to(self.device)
            expert_result = expert_weight * expert_result
            results.append(expert_result)
        # Combining results
        out = torch.stack(results, dim=0).sum(dim=0)

        if output_only:
            return out
        else:
            return {'output': out,
                    'aux_loss': aux_loss,
                    'examples_per_expert': examples_per_expert,
                    'expert_importance': expert_importance,
                    'weights': weights}

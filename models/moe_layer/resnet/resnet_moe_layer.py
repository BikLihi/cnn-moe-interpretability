from models.moe_layer.base_components.base_moe_layer import BaseMoELayer
from models.moe_layer.resnet.resnet_expert import ResNetExpert


class ResNetMoeLayer(BaseMoELayer):
    def __init__(self, in_channels, out_channels, gate, num_experts=8, top_k=2, num_blocks=1, use_noise=False, ):
        
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            gate=gate,
            num_experts=num_experts,
            top_k=top_k)

        if self.num_experts == 1:
            self.gate.set_parameter_requires_grad(False)

        for i in range(self.num_experts):
            expert = ResNetExpert(
                in_channels=in_channels,
                out_channels=out_channels,
                name='Expert ' + str(i),
                num_blocks=num_blocks)
            self.experts.append(expert)

    def forward(self, x, output_only=True):
        return super().forward(x, output_only)

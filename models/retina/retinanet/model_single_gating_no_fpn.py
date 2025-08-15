import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from models.retina.retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from models.retina.retinanet.anchors import Anchors
from losses.importance_loss import importance
from models.retina.retinanet import retina_losses
from models.moe_layer.base_components.base_moe_layer import BaseMoELayer

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        # P4_x = self.P4_1(C4)
        # P4_x = P5_upsampled_x + P4_x
        # P4_upsampled_x = self.P4_upsampled(P4_x)
        # P4_x = self.P4_2(P4_x)

        # P3_x = self.P3_1(C3)
        # P3_x = P3_x + P4_upsampled_x
        # P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x, raw_output=False):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        if raw_output:
            return out

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        out = out.contiguous().view(out.shape[0], -1, 4)
        return out


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x, raw_output=False):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        if raw_output:
            return out

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = retina_losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs, return_expert_predictions=False):
        self.examples_per_expert = None
        self.expert_importance = None
        self.weights = None


        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        aux_losses = None

        if type(self.regressionModel) == RegressionModelMoE:
            moe_outputs = [self.regressionModel(feature, output_only=False) for feature in features]
        else:
            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        if type(self.classificationModel) == ClassificationModelMoE:
            moe_outputs = [self.classificationModel(feature, output_only=False) for feature in features]
        else:
            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        

def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model

class RegressionModelMoE(BaseMoELayer):
    def __init__(self, num_features_in, num_experts, top_k, gating_network, num_anchors=9, feature_size=256):
        self.num_anchors = num_anchors
        experts = nn.ModuleList([RegressionModel(num_features_in, num_anchors, feature_size) for i in range(num_experts)])
        super().__init__(
            gate=gating_network,
            num_experts=num_experts,
            top_k=top_k,
            experts=experts)

        if self.num_experts == 1:
            self.gate.set_parameter_requires_grad(False)


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
                    'expert_importance': torch.tensor([1.0])}

        weights = self.gate.compute_gating(x)
        examples_per_expert = (weights > 0).sum(dim=0)
        expert_importance = importance(weights)

        aux_loss = self.gate.compute_loss(weights)
        mask = weights > 0
        results = []
        results_before_weighting = []
        for i in range(self.num_experts):
            # select mask according to computed gates (conditional computing)
            mask_expert = mask[:, i]
            # apply mask to inputs
            expert_input = x[mask_expert]
            # compute outputs for selected examples
            if expert_input.shape[0] == 0:
                expert_output = torch.zeros([x.size()[0], x.size()[2] * x.size()[3] * self.num_anchors, 4], device=self.device)
            else:
                expert_output = self.experts[i](expert_input, raw_output=False).to(self.device)
            # calculate output shape
            output_shape = list(expert_output.shape)
            output_shape[0] = x.size()[0]
            # store expert results in matrix
            expert_result = torch.zeros(output_shape, device=self.device)
            if torch.sum(mask_expert) > 0:
                expert_result[mask_expert] = expert_output

            # weight expert's results
            expert_weight = weights[:, i].reshape(
                expert_result.shape[0], 1, 1).to(self.device)
            results_before_weighting.append(expert_result)
            expert_result = expert_weight * expert_result
            results.append(expert_result)
        # Combining results
        out = torch.stack(results, dim=0).sum(dim=0)
        if weights.shape[0] == 1:
            weights = weights.expand(out.shape[1], 4)
        if output_only:
            return out
        else:
            return {'output': out,
                    'aux_loss': aux_loss,
                    'examples_per_expert': examples_per_expert,
                    'expert_importance': expert_importance,
                    'weights': weights,
                    'expert_predictions': results_before_weighting}


class ClassificationModelMoE(BaseMoELayer):
    def __init__(self, num_features_in, num_experts, top_k, gating_network, num_classes, num_anchors=9, feature_size=256):
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        experts = nn.ModuleList([ClassificationModel(num_features_in, num_anchors, num_classes) for i in range(num_experts)])
        super().__init__(
            gate=gating_network,
            num_experts=num_experts,
            top_k=top_k,
            experts=experts)

        if self.num_experts == 1:
            self.gate.set_parameter_requires_grad(False)


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
                    'expert_importance': torch.tensor([1.0])}

        weights = self.gate.compute_gating(x)
        examples_per_expert = (weights > 0).sum(dim=0)
        expert_importance = importance(weights)

        aux_loss = 0.0
        mask = weights > 0
        results = []
        for i in range(self.num_experts):
            # select mask according to computed gates (conditional computing)
            mask_expert = mask[:, i]
            # apply mask to inputs
            expert_input = x[mask_expert]
            # compute outputs for selected examples
            if expert_input.shape[0] == 0:
                expert_output = torch.zeros([x.size()[0], x.size()[2] * x.size()[3] * self.num_anchors, self.num_classes], device=self.device)
            else:
                expert_output = self.experts[i](expert_input, raw_output=False).to(self.device)
            # calculate output shape
            output_shape = list(expert_output.shape)
            output_shape[0] = x.size()[0]
            # store expert results in matrix
            expert_result = torch.zeros(output_shape, device=self.device)
            if torch.sum(mask_expert) > 0:
                expert_result[mask_expert] = expert_output

            # weight expert's results
            expert_weight = weights[:, i].reshape(
                expert_result.shape[0], 1, 1).to(self.device)
            expert_result = expert_weight * expert_result
            results.append(expert_result)
        # Combining results
        out = torch.stack(results, dim=0).sum(dim=0)
        if weights.shape[0] == 1:
            weights = weights.expand(out.shape[1], self.num_experts)
        if output_only:
            return out
        else:
            return {'output': out,
                    'aux_loss': aux_loss,
                    'examples_per_expert': examples_per_expert,
                    'expert_importance': expert_importance,
                    'weights': weights}


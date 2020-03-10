import torch
import torch.nn as nn
from torchvision import models

from conf import core, model


class FineTuneModel(nn.Module):
    def __init__(self, num_classes, freeze=True):
        super(FineTuneModel, self).__init__()
        self.base = _base_model()

        for param in self.parameters():
            param.requires_grad_(not freeze)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)


class SideTuneModel(nn.Module):
    def __init__(self, num_classes):
        super(SideTuneModel, self).__init__()
        self.base = _base_model()
        # Lock base model parameters
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.side = _side_model()
        self.side.fc = nn.Linear(self.side.fc.in_features, num_classes)
        self.merge = _merge

        if core.use_gpu:
            self.base.cuda()
            self.side.cuda()

    def forward(self, x):
        s_x = x.clone()
        if core.use_gpu:
            s_x.cuda()

        # Start of the base model forward
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        b_x = self.base.avgpool(x)
        # End of the base model forward

        # Start of the side model forward
        s_x = self.side.conv1(s_x)
        s_x = self.side.bn1(s_x)
        s_x = self.side.relu(s_x)
        s_x = self.side.maxpool(s_x)

        s_x = self.side.layer1(s_x)
        s_x = self.side.layer2(s_x)
        s_x = self.side.layer3(s_x)
        s_x = self.side.layer4(s_x)

        s_x = self.side.avgpool(s_x)
        # End of the side model forward

        x_merge = self.merge(b_x, s_x)
        x_merge = self.side.fc(x_merge.squeeze())  # get rid of the useless dimensions

        return x_merge


def _merge(base_encoding, side_encoding):
    weights = [model.image_alpha, 1 - model.image_alpha]
    outputs_to_merge = [base_encoding] + [side_encoding]
    merged_encoding = torch.zeros_like(base_encoding)
    if core.use_gpu:
        merged_encoding.cuda()

    for a, out in zip(weights, outputs_to_merge):
        merged_encoding += a * out
    return merged_encoding


def _base_model():
    return models.resnet50(pretrained=True)


def _side_model():
    return models.resnet50(pretrained=True)

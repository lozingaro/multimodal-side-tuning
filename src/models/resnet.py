import torch
import torch.nn as nn
from torchvision import models

from conf import core


class FineTuneModel(nn.Module):
    def __init__(self, num_classes):
        super(FineTuneModel, self).__init__()
        self.base = models.resnet50(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        self.base = self.base.to(core.device)

    def forward(self, x):
        x = x.to(core.device)

        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.fc(x)

        return x


class SideTuneModel(nn.Module):
    def __init__(self, num_classes, alpha=.5):
        super(SideTuneModel, self).__init__()
        self.alpha = alpha
        self.base = models.resnet50(pretrained=True)
        # Lock base model parameters
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.base = self.base.to(core.device)

        self.side = models.resnet50(pretrained=True)
        self.side.fc = nn.Linear(self.side.fc.in_features, num_classes)
        self.side = self.side.to(core.device)

        self.merge = self._merge

    def forward(self, x):
        x = x.to(core.device)
        s_x = x.clone()
        s_x = s_x.to(core.device)

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

    def _merge(self, base_encoding, side_encoding):
        weights = [self.alpha, 1 - self.alpha]
        outputs_to_merge = [base_encoding] + [side_encoding]
        merged_encoding = torch.zeros_like(base_encoding)
        if core.use_gpu:
            merged_encoding.cuda()

        for a, out in zip(weights, outputs_to_merge):
            merged_encoding += a * out
        return merged_encoding

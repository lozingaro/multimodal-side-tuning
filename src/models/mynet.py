import torch
import torch.nn as nn
from torchvision import models
from .utils import merge

# TODO create side network custom
class SideTuneModel(nn.Module):
    def __init__(self, num_classes, alpha=.5):
        super(SideTuneModel, self).__init__()
        self.alpha = alpha
        self.base = models.mobilenet_v2(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.side = models.mobilenet_v2(pretrained=True)
        self.side.classifier[1] = nn.Linear(self.side.last_channel, num_classes)
        self.merge = merge
        self.side.fc = torch.nn.Linear(self.side.fc.in_features, 300)
        # concatenate
        # Linear(600, num_classes)

    def forward(self, x):
        s_x = x.clone()

        b_x = self.base.features(x)
        s_x = self.side.features(s_x)

        x_merge = self.merge(self.alpha, b_x, s_x)
        x_merge = x_merge.mean([2, 3])
        x_merge = self.side.classifier(x_merge)

        return x_merge

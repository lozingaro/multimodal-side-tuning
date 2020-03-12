import torch.nn as nn
from torchvision import models

from conf import core


class FineTuneModel(nn.Module):
    def __init__(self, num_classes):
        super(FineTuneModel, self).__init__()
        self.base = models.mobilenet_v2(pretrained=True)
        self.base.classifier[1] = nn.Linear(self.base.last_channel, num_classes)
        self.base = self.base.to(core.device)

    def forward(self, x):
        x = x.to(core.device)
        
        x = self.base.features(x)
        x = x.mean([2, 3])
        x = self.base.classifier(x)
        return x

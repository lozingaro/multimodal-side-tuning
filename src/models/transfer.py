import torch.nn as nn
from torchvision import models


class FineTuneModel(nn.Module):
    def __init__(self, out_features, pretrained=True):
        super(FineTuneModel, self).__init__()
        self.out_features = out_features
        self.pretrained = pretrained
        self.base = self._base_model()
        self.in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(self.in_features, out_features)

    def forward(self, x):
        return self.base.forward(x)

    def _base_model(self):
        # return models.mobilenet_v2(pretrained=self.pretrained)
        return models.resnet18(pretrained=self.pretrained)

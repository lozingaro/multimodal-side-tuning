import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .utils import merge


class AgneseNetModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, in_channels=1, out_channels=512, kernel_sizes=None, stride=2,
                 dropout_prob=.5, num_classes=10):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [250, 125, 62]

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(i, embedding_dim),
            # stride=stride
        ) for i in kernel_sizes])
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(out_channels * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embeddings(x)
        x = torch.unsqueeze(x, 1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MobileNetV2SideTuneModel(nn.Module):
    def __init__(self, num_classes, alpha=.5):
        super(MobileNetV2SideTuneModel, self).__init__()
        self.alpha = alpha
        self.base = torchvision.models.mobilenet_v2(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.side = torchvision.models.mobilenet_v2(pretrained=True)
        self.side.classifier[1] = nn.Linear(self.side.last_channel, num_classes)
        self.merge = merge

    def forward(self, x):
        s_x = x.clone()

        b_x = self.base.features(x)
        s_x = self.side.features(s_x)

        x_merge = self.merge(self.alpha, b_x, s_x)
        x_merge = x_merge.mean([2, 3])
        x_merge = self.side.classifier(x_merge)

        return x_merge


class ReseNetSideTuneModel(nn.Module):
    def __init__(self, num_classes, alpha=.5):
        super(ReseNetSideTuneModel, self).__init__()
        self.alpha = alpha
        self.base = torchvision.models.resnet50(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.side = torchvision.models.resnet50(pretrained=True)
        self.side.fc = nn.Linear(self.side.fc.in_features, num_classes)
        self.merge = merge

    def forward(self, x):
        s_x = x.clone()

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

        x_merge = self.merge(self.alpha, b_x, s_x)
        x_merge = torch.flatten(x_merge, 1)
        x_merge = self.side.fc(x_merge)

        return x_merge

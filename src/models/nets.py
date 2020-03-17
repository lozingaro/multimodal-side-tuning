import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .utils import merge


class ShawnNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_classes, dropout=.5):
        super(ShawnNet, self).__init__()
        V = vocab_size
        D = embedding_dim
        C = num_classes
        Ci = 1
        Co = 512
        Ks = [3, 4, 5]

        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc(x)

        return logit


class AgneseNetV2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, dropout=.5, alpha=.5):
        super(AgneseNetV2, self).__init__()
        self.alpha = alpha
        self.merge = merge

        self.base = torchvision.models.mobilenet_v2(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)

        Ks = [3, 4, 5]
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, 512, (K, embedding_dim)) for K in Ks])
        self.dropout = nn.Dropout(.2)
        self.fc1 = nn.Linear(len(Ks) * 512, self.base.last_channel)  # check on mobile net output
        self.fc2 = nn.Linear(self.base.last_channel, num_classes)

    def forward(self, b_x, s_x):
        s_x = self.embed(s_x)
        s_x = s_x.unsqueeze(1)
        s_x = [F.relu(conv(s_x)).squeeze(3) for conv in self.convs]
        s_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in s_x]
        s_x = torch.cat(s_x, 1)
        s_x = self.fc1(s_x)

        b_x = self.base.features(b_x)
        b_x = b_x.mean([2, 3])
        x = self.merge((0.5 + 0.75) / 2, b_x, s_x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class AgneseNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, dropout=.5, alpha=.5):
        super(AgneseNet, self).__init__()
        self.alpha = alpha
        self.merge = merge

        self.base = torchvision.models.mobilenet_v2(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)

        Ks = [3, 4, 5]
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, 512, (K, embedding_dim)) for K in Ks])
        self.dropout = nn.Dropout(.2)
        self.fc1 = nn.Linear(len(Ks) * 512, self.base.last_channel)  # check on mobile net output
        self.fc2 = nn.Linear(self.base.last_channel, num_classes)

    def forward(self, b_x, s_x):
        s_x = self.embed(s_x)
        s_x = s_x.unsqueeze(1)
        s_x = [F.relu(conv(s_x)).squeeze(3) for conv in self.convs]
        s_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in s_x]
        s_x = torch.cat(s_x, 1)
        s_x = self.fc1(s_x)

        b_x = self.base.features(b_x)
        b_x = b_x.mean([2, 3])
        x = self.merge((0.5 + 0.75) / 2, b_x, s_x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CedricNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes=10):
        super(CedricNet, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(1, 512, (12, 300))
        self.conv2 = nn.Conv1d(512, 512, 12)
        self.dropout = nn.Dropout(.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embeddings(x)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = F.relu(x)
        x = x.squeeze(3)
        x = F.max_pool1d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool1d(x, 2)

        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MobileNetV2Savona(nn.Module):
    def __init__(self, num_classes, alpha=.5):
        super(MobileNetV2Savona, self).__init__()
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


class ReseNetSavona(nn.Module):
    def __init__(self, num_classes, alpha=.5):
        super(ReseNetSavona, self).__init__()
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

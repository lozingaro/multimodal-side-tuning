import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .utils import merge


class ShawnNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, ch_input=1, num_filters=512, windows=None, dropout_prob=.5,
                 num_classes=10):
        super(ShawnNet, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.ch_input = ch_input
        self.num_filters = num_filters
        if windows is None:
            self.windows = [3, 4, 5]
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes

        self.embed1 = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.conv2a = nn.Conv2d(self.ch_input, self.num_filters, (self.windows[0], self.embedding_dim))
        self.conv2b = nn.Conv2d(self.ch_input, self.num_filters, (self.windows[1], self.embedding_dim))
        self.conv2c = nn.Conv2d(self.ch_input, self.num_filters, (self.windows[2], self.embedding_dim))
        self.dropout3 = nn.Dropout(self.dropout_prob)
        self.fc4 = nn.Linear(len(self.windows) * self.num_filters, self.num_classes)

    def forward(self, x):
        x = self.embed1(x)
        x = x.unsqueeze(1)

        x2a = self.conv2a(x)
        x2a = F.relu(x2a).squeeze(3)
        x2a = F.max_pool1d(x2a, x2a.size(2)).squeeze(2)

        x2b = self.conv2b(x)
        x2b = F.relu(x2b).squeeze(3)
        x2b = F.max_pool1d(x2b, x2b.size(2)).squeeze(2)

        x2c = self.conv2c(x)
        x2c = F.relu(x2c).squeeze(3)
        x2c = F.max_pool1d(x2c, x2c.size(2)).squeeze(2)

        x = torch.cat((x2a, x2b, x2c), 1)

        x = self.dropout3(x)
        x = self.fc4(x)

        return x


class TextSideNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, alpha=.5):
        super(TextSideNet, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.merge = merge

        self.base = torchvision.models.mobilenet_v2(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.side = ShawnNet(self.vocab_size, self.embedding_dim)

        self.fc1fus = nn.Linear(self.side.num_filters * len(self.side.windows), 1280)
        # self.fc2fus = nn.Linear(self.base.last_channel, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc3fus = nn.Linear(1280, self.num_classes)

    def forward(self, b_x, s_x):
        s_x = self.side.embed1(s_x)
        s_x = s_x.unsqueeze(1)

        s_x2a = self.side.conv2a(s_x)
        s_x2a = F.relu(s_x2a).squeeze(3)
        s_x2a = F.max_pool1d(s_x2a, s_x2a.size(2)).squeeze(2)

        s_x2b = self.side.conv2b(s_x)
        s_x2b = F.relu(s_x2b).squeeze(3)
        s_x2b = F.max_pool1d(s_x2b, s_x2b.size(2)).squeeze(2)

        s_x2c = self.side.conv2c(s_x)
        s_x2c = F.relu(s_x2c).squeeze(3)
        s_x2c = F.max_pool1d(s_x2c, s_x2c.size(2)).squeeze(2)

        s_x = torch.cat((s_x2a, s_x2b, s_x2c), 1)
        s_x = self.fc1fus(s_x)

        b_x = self.base.features(b_x)
        b_x = b_x.mean([2, 3])
        # ---------- locked ----------
        # b_x = self.fc2fus(b_x)

        x, ds = self.merge(self.alpha, b_x, s_x)
        x = self.dropout(x)
        x = self.fc3fus(x)

        return x, ds


class TextSideNet_sideFC(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, alpha=.5):
        super(TextSideNet_sideFC, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.merge = merge

        self.base = torchvision.models.mobilenet_v2(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.side = ShawnNet(self.vocab_size, self.embedding_dim)

        self.fc1fus = nn.Linear(self.side.num_filters * len(self.side.windows), 128)
        self.fc2fus = nn.Linear(self.base.last_channel, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc3fus = nn.Linear(128, self.num_classes)

    def forward(self, b_x, s_x):
        s_x = self.side.embed1(s_x)
        s_x = s_x.unsqueeze(1)

        s_x2a = self.side.conv2a(s_x)
        s_x2a = F.relu(s_x2a).squeeze(3)
        s_x2a = F.max_pool1d(s_x2a, s_x2a.size(2)).squeeze(2)

        s_x2b = self.side.conv2b(s_x)
        s_x2b = F.relu(s_x2b).squeeze(3)
        s_x2b = F.max_pool1d(s_x2b, s_x2b.size(2)).squeeze(2)

        s_x2c = self.side.conv2c(s_x)
        s_x2c = F.relu(s_x2c).squeeze(3)
        s_x2c = F.max_pool1d(s_x2c, s_x2c.size(2)).squeeze(2)

        s_x = torch.cat((s_x2a, s_x2b, s_x2c), 1)
        s_x = self.fc1fus(s_x)

        b_x = self.base.features(b_x)
        # ---------- locked ----------
        b_x = F.avg_pool2d(b_x, b_x.size(2))
        b_x = b_x.squeeze()
        b_x = self.fc2fus(b_x)

        x, ds = self.merge(self.alpha, b_x, s_x)
        x = self.dropout(x)
        x = self.fc3fus(x)

        return x, ds


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

        x_merge, _ = self.merge(self.alpha, b_x, s_x)
        x_merge = x_merge.mean([2, 3])
        x_merge = self.side.classifier(x_merge)

        return x_merge


class ResNetSavona(nn.Module):
    def __init__(self, num_classes, alpha=.5):
        super(ResNetSavona, self).__init__()
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

        x_merge, _ = self.merge(self.alpha, b_x, s_x)
        x_merge = torch.flatten(x_merge, 1)
        x_merge = self.side.fc(x_merge)

        return x_merge

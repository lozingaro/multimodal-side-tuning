import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .utils import merge


class TextImageSideNetBaseFC(nn.Module):
    def __init__(self, embedding_dim, num_classes, alphas=None,
                 dropout_prob=.2, custom_embedding=False,
                 custom_num_embeddings=0):
        super(TextImageSideNetBaseFC, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        if alphas is None:
            alphas = [.3, .3]
        self.alphas = alphas
        self.dropout_prob = dropout_prob

        self.base = torchvision.models.mobilenet_v2(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.side_image = torchvision.models.mobilenet_v2(pretrained=True)
        self.side_text = ShawnNet(self.embedding_dim,
                                  custom_embedding=custom_embedding,
                                  custom_num_embeddings=custom_num_embeddings)

        self.fc1fus = nn.Sequential(nn.Dropout(self.dropout_prob),
                                    nn.Linear(self.side_text.num_filters * len(
                                        self.side_text.windows), 128))
        self.fc2fus = nn.Sequential(nn.Dropout(self.dropout_prob),
                                    nn.Linear(self.base.last_channel, 128))
        self.fc3fus = nn.Sequential(nn.Dropout(self.dropout_prob),
                                    nn.Linear(self.side_image.last_channel,
                                              128))

        self.classifier = nn.Linear(128, self.num_classes)

    def forward(self, y):
        b_x, s_text_x = y[0], y[1]

        if self.side_text.custom_embedding:
            s_text_x = self.side_text.embedding(s_text_x)
        s_text_x = s_text_x.unsqueeze(1)

        s_text_x = [F.relu(conv(s_text_x)).squeeze(3) for conv in
                    self.side_text.convs]
        s_text_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in s_text_x]
        s_text_x = torch.cat(s_text_x, 1)
        s_text_x = self.fc1fus(s_text_x)

        s_image_x = b_x.clone()

        b_x = self.base.features(b_x)
        b_x = b_x.mean([2, 3])
        b_x = self.fc2fus(b_x)

        s_image_x = self.side_image.features(s_image_x)
        s_image_x = s_image_x.mean([2, 3])
        s_image_x = self.fc3fus(s_image_x)

        x, d = merge([b_x, s_image_x, s_text_x], self.alphas,
                     return_distance=True)
        x = self.classifier(x)

        return x, d


class TextImageSideNet(nn.Module):
    def __init__(self, embedding_dim, num_classes, alphas=None,
                 dropout_prob=.2, custom_embedding=False,
                 custom_num_embeddings=0):
        super(TextImageSideNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        if alphas is None:
            alphas = [.3, .3]
        self.alphas = alphas
        self.dropout_prob = dropout_prob

        self.base = torchvision.models.mobilenet_v2(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.side_image = torchvision.models.mobilenet_v2(pretrained=True)
        self.side_text = ShawnNet(self.embedding_dim,
                                  custom_embedding=custom_embedding,
                                  custom_num_embeddings=custom_num_embeddings)

        self.fc1fus = nn.Linear(
            self.side_text.num_filters * len(self.side_text.windows),
            self.base.last_channel)
        self.classifier = nn.Sequential(nn.Dropout(self.dropout_prob),
                                        nn.Linear(self.base.last_channel, self.num_classes))

    def forward(self, y):
        b_x, s_text_x = y[0], y[1]

        if self.side_text.custom_embedding:
            s_text_x = self.side_text.embedding(s_text_x)
        s_text_x = s_text_x.unsqueeze(1)

        s_text_x = [F.relu(conv(s_text_x)).squeeze(3) for conv in
                    self.side_text.convs]
        s_text_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in s_text_x]
        s_text_x = torch.cat(s_text_x, 1)
        s_text_x = self.fc1fus(s_text_x)

        s_image_x = b_x.clone()

        b_x = self.base.features(b_x)
        b_x = b_x.mean([2, 3])

        s_image_x = self.side_image.features(s_image_x)
        s_image_x = s_image_x.mean([2, 3])

        x, d = merge([b_x, s_image_x, s_text_x], self.alphas,
                     return_distance=True)
        x = self.classifier(x)

        return x, d


class TextImageSideNetSideFC(nn.Module):
    def __init__(self, embedding_dim, num_classes, alphas=None,
                 dropout_prob=.2, custom_embedding=False,
                 custom_num_embeddings=0, side_fc=512):
        super(TextImageSideNetSideFC, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        if alphas is None:
            alphas = [.3, .3]
        self.alphas = alphas
        self.dropout_prob = dropout_prob

        self.base = torchvision.models.mobilenet_v2(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.side_image = torchvision.models.mobilenet_v2(pretrained=True)
        self.side_text = ShawnNet(self.embedding_dim,
                                  custom_embedding=custom_embedding,
                                  custom_num_embeddings=custom_num_embeddings)

        self.fc1fus = nn.Linear(
            self.side_text.num_filters * len(self.side_text.windows),
            self.base.last_channel)
        self.classifier = nn.Sequential(nn.Dropout(self.dropout_prob),
                                        nn.Linear(self.base.last_channel, side_fc),
                                        nn.Dropout(self.dropout_prob),
                                        nn.Linear(side_fc, self.num_classes))

    def forward(self, y):
        b_x, s_text_x = y[0], y[1]

        if self.side_text.custom_embedding:
            s_text_x = self.side_text.embedding(s_text_x)
        s_text_x = s_text_x.unsqueeze(1)

        s_text_x = [F.relu(conv(s_text_x)).squeeze(3) for conv in
                    self.side_text.convs]
        s_text_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in s_text_x]
        s_text_x = torch.cat(s_text_x, 1)
        s_text_x = self.fc1fus(s_text_x)

        s_image_x = b_x.clone()

        b_x = self.base.features(b_x)
        b_x = b_x.mean([2, 3])

        s_image_x = self.side_image.features(s_image_x)
        s_image_x = s_image_x.mean([2, 3])

        x, d = merge([b_x, s_image_x, s_text_x], self.alphas,
                     return_distance=True)
        x = self.classifier(x)

        return x, d


class TextSideResNet(nn.Module):
    def __init__(self, embedding_dim, num_classes, alpha=.5,
                 dropout_prob=.2):
        super(TextSideResNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.dropout_prob = dropout_prob

        self.base = torchvision.models.resnet50(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.side = ShawnNet(self.embedding_dim, windows=[2, 3, 4, 5])

        self.classifier = nn.Sequential(nn.Dropout(self.dropout_prob),
                                        nn.Linear(2048, self.num_classes))

    def forward(self, y):
        b_x, s_x = y[0], y[1]

        s_x = s_x.unsqueeze(1)
        s_x = [F.relu(conv(s_x)).squeeze(3) for conv in self.side.convs]
        s_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in s_x]
        s_x = torch.cat(s_x, 1)

        b_x = self.base.conv1(b_x)  # resnet forward
        b_x = self.base.bn1(b_x)
        b_x = self.base.relu(b_x)
        b_x = self.base.maxpool(b_x)
        b_x = self.base.layer1(b_x)
        b_x = self.base.layer2(b_x)
        b_x = self.base.layer3(b_x)
        b_x = self.base.layer4(b_x)
        b_x = self.base.avgpool(b_x)
        b_x = torch.flatten(b_x, 1)

        x = merge([b_x, s_x], [self.alpha])
        x = self.classifier(x)

        return x


class TextSideNet(nn.Module):
    def __init__(self, embedding_dim, num_classes, alpha=.5,
                 dropout_prob=.2):
        super(TextSideNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.dropout_prob = dropout_prob

        self.base = torchvision.models.mobilenet_v2(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.side = ShawnNet(self.embedding_dim)

        self.fc1fus = nn.Linear(self.side.num_filters * len(self.side.windows),
                                self.base.last_channel)
        self.classifier = nn.Sequential(nn.Dropout(self.dropout_prob),
                                        nn.Linear(self.base.last_channel,
                                                  self.num_classes))

    def forward(self, y):
        b_x, s_x = y[0], y[1]

        b_x = self.base.features(b_x)
        b_x = b_x.mean([2, 3])

        s_x = s_x.unsqueeze(1)
        s_x = [F.relu(conv(s_x)).squeeze(3) for conv in self.side.convs]
        s_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in s_x]
        s_x = torch.cat(s_x, 1)
        s_x = self.fc1fus(s_x)

        x, d = merge([b_x, s_x], [self.alpha], return_distance=True)
        x = self.classifier(x)

        return x, d


class TextSideNetBaseFC(nn.Module):
    def __init__(self, embedding_dim, num_classes=10, alpha=.5,
                 dropout_prob=.2):
        super(TextSideNetBaseFC, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.dropout_prob = dropout_prob

        self.base = torchvision.models.mobilenet_v2(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.side = ShawnNet(self.embedding_dim)

        self.fc1fus = nn.Linear(self.side.num_filters * len(self.side.windows),
                                128)
        self.fc2fus = nn.Linear(self.base.last_channel, 128)

        self.classifier = nn.Sequential(nn.Dropout(self.dropout_prob),
                                        nn.Linear(128, self.num_classes))

    def forward(self, y):
        b_x, s_x = y[0], y[1]

        s_x = s_x.unsqueeze(1)

        s_x = [F.relu(conv(s_x)).squeeze(3) for conv in self.side.convs]
        s_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in s_x]
        s_x = torch.cat(s_x, 1)
        s_x = self.fc1fus(s_x)

        b_x = self.base.features(b_x)
        b_x = F.avg_pool2d(b_x, b_x.size(2))
        b_x = b_x.squeeze()
        b_x = self.fc2fus(b_x)

        x = merge([b_x, s_x], [self.alpha])
        x = self.classifier(x)

        return x


class MobileNet(nn.Module):
    def __init__(self, num_classes, alpha=.5, dropout_prob=.2):
        super(MobileNet, self).__init__()
        self.alpha = alpha
        self.dropout_prob = dropout_prob

        self.base = torchvision.models.mobilenet_v2(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.side = torchvision.models.mobilenet_v2(pretrained=True)
        self.classifier = nn.Sequential(nn.Dropout(self.dropout_prob),
                                        nn.Linear(self.side.last_channel,
                                                  num_classes))

    def forward(self, x):
        s_x = x.clone()

        b_x = self.base.features(x)
        s_x = self.side.features(s_x)

        x, d = merge([b_x, s_x], [self.alpha], return_distance=True)
        x = x.mean([2, 3])
        x = self.classifier(x)

        return x, d


class ResNet(nn.Module):
    def __init__(self, num_classes, alpha=.5, dropout_prob=.2):
        super(ResNet, self).__init__()
        self.alpha = alpha
        self.dropout_prob = dropout_prob
        self.base = torchvision.models.resnet50(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.side = torchvision.models.resnet50(pretrained=True)
        self.classifier = nn.Sequential(nn.Dropout(self.dropout_prob),
                                        nn.Linear(self.side.fc.in_features,
                                                  num_classes))

    def forward(self, x):
        s_x = x.clone()

        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        b_x = self.base.avgpool(x)

        s_x = self.side.conv1(s_x)
        s_x = self.side.bn1(s_x)
        s_x = self.side.relu(s_x)
        s_x = self.side.maxpool(s_x)

        s_x = self.side.layer1(s_x)
        s_x = self.side.layer2(s_x)
        s_x = self.side.layer3(s_x)
        s_x = self.side.layer4(s_x)

        s_x = self.side.avgpool(s_x)

        x = merge([b_x, s_x], [self.alpha])
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class ShawnNet(nn.Module):
    def __init__(self, embedding_dim, num_filters=512, windows=None,
                 dropout_prob=.2, num_classes=10,
                 custom_embedding=False, custom_num_embeddings=0):
        super(ShawnNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        if windows is None:
            self.windows = [3, 4, 5]
        else:
            self.windows = windows
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        self.custom_embedding = custom_embedding

        if self.custom_embedding:
            self.embedding = nn.Embedding(custom_num_embeddings, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (i, self.embedding_dim)) for i in
            self.windows
        ])
        self.classifier = nn.Sequential(nn.Dropout(self.dropout_prob),
                                        nn.Linear(len(
                                            self.windows) * self.num_filters,
                                                  self.num_classes))

    def forward(self, x):
        if self.custom_embedding:
            x = self.embedding(x)
        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.classifier(x)

        return x

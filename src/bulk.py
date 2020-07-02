from __future__ import division, print_function

import random
from warnings import filterwarnings

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.tobacco import TobaccoDataset
from models import TrainingPipeline, FusionSideNetFc, FusionNetConcat, FusionSideNetDirect, FusionSideNetFcResNet

filterwarnings("ignore")
cudnn.deterministic = True
cudnn.benchmark = False

for task in conf.tasks:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    d = TobaccoDataset(conf.tobacco_img_root_dir, conf.tobacco_txt_root_dir)
    r = torch.utils.data.random_split(d, [800, 200, 2482])
    d_train = r[0]
    d_val = r[1]
    d_test = r[2]
    dl_train = DataLoader(d_train, batch_size=16, shuffle=True)
    dl_val = DataLoader(d_val, batch_size=4, shuffle=True)
    dl_test = DataLoader(d_test, batch_size=32, shuffle=False)

    if task[0] == 'concat':
        model = FusionNetConcat(300,
                                num_classes=10,
                                dropout_prob=.5).to(conf.device)
    elif task[0] == 'mobilenet':
        model = FusionSideNetFc(300,
                                num_classes=10,
                                alphas=[int(i) / 10 for i in task[4].split('-')],
                                dropout_prob=.5,
                                side_fc=1024).to(conf.device)
    elif task[0] == 'resnet':
        model = FusionSideNetFcResNet(300,
                                      num_classes=10,
                                      alphas=[int(i) / 10 for i in task[4].split('-')],
                                      dropout_prob=.5,
                                      side_fc=1024).to(conf.device)
    else:
        model = FusionSideNetDirect(300,
                                    num_classes=10,
                                    alphas=[int(i) / 10 for i in task[4].split('-')],
                                    dropout_prob=.5).to(conf.device)

    if task[3] == 'min':
        _, c = np.unique(np.array(d.targets)[d_train.indices], return_counts=True)
        weight = torch.from_numpy(np.min(c) / c).type(torch.FloatTensor).to(conf.device)
    elif task[3] == 'max':
        _, c = np.unique(np.array(d.targets)[d_train.indices], return_counts=True)
        weight = torch.from_numpy(np.max(c) / c).type(torch.FloatTensor).to(conf.device)
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight).to(conf.device)

    if task[1] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        scheduler = None
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .1 * (1.0 - float(epoch) / 100.0) ** .5)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    pipeline = TrainingPipeline(model, criterion, optimizer, scheduler, device=conf.device, num_classes=10)
    best_valid_acc, test_acc, confusion_matrix = pipeline.run(dl_train, dl_val, dl_test, num_epochs=100)

    s = f'{",".join([str(i) for i in task])},' \
        f'{best_valid_acc:.3f},' \
        f'{test_acc:.3f},' \
        f'{",".join([f"{r[i] / np.sum(r):.3f}" for i, r in enumerate(confusion_matrix)])}\n'
    with open('../test/results_tobacco.csv', 'a+') as f:
        f.write(s)

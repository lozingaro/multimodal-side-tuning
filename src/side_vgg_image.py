from __future__ import division, print_function

import random
from warnings import filterwarnings

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.tobacco import TobaccoImgDataset
from models import TrainingPipeline, SideNetVGG

filterwarnings("ignore")
cudnn.deterministic = True
cudnn.benchmark = False

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

d = TobaccoImgDataset(conf.tobacco_img_root_dir)
d_train, d_val, d_test = torch.utils.data.random_split(d, [800, 200, 2482])
dl_train = DataLoader(d_train, batch_size=16, shuffle=True)
dl_val = DataLoader(d_val, batch_size=4, shuffle=True)
dl_test = DataLoader(d_test, batch_size=32, shuffle=False)
num_classes = 10
num_epochs = 100

model = SideNetVGG(num_classes=num_classes,
                   alphas=[.5, .5],
                   dropout_prob=.5).to(conf.device)

criterion = nn.CrossEntropyLoss().to(conf.device)
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda epoch: .1 * (1.0 - float(epoch) / float(num_epochs)) ** .5
)
pipeline = TrainingPipeline(model, criterion, optimizer, scheduler, device=conf.device, num_classes=num_classes)
best_valid_acc, test_acc, confusion_matrix = pipeline.run(dl_train, dl_val, dl_test, num_epochs=num_epochs)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))
s = 'side-vgg,sgd,-,no,5-5,' \
    f'{best_valid_acc:.3f},' \
    f'{test_acc:.3f},' \
    f'{",".join([f"{r[i] / np.sum(r):.3f}" for i, r in enumerate(confusion_matrix)])}\n'
with open('../test/results_tobacco.csv', 'a+') as f:
    f.write(s)

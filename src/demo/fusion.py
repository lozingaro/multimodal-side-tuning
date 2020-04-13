from __future__ import division, print_function

import random
from warnings import filterwarnings

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.rvl_cdip import RvlDataset
from datasets.tobacco import TobaccoDataset
from models import TrainingPipeline, FusionSideNetFc, FusionNetConcat, FusionSideNetDirect

filterwarnings("ignore")
cudnn.deterministic = True
cudnn.benchmark = False
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

model = FusionSideNetFc(300,
                        num_classes=10,
                        alphas=[.3, .3, .4],
                        dropout_prob=.5,
                        side_fc=256).to(conf.core.device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
_, c = np.unique(np.array(d.targets)[d_train.indices], return_counts=True)
weight = torch.from_numpy(np.min(c) / c).type(torch.FloatTensor).to(conf.device)
criterion = nn.CrossEntropyLoss(weight=weight).to(conf.device)
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .1 * (1.0 - float(epoch) / 10.0) ** .5)
pipeline = TrainingPipeline(model, criterion, optimizer, scheduler, device=conf.device, num_classes=10)
best_valid_acc, test_acc, cm = pipeline.run(dl_train, dl_val, dl_test, num_epochs=10)

s = f'1280x512x10,sgd,fasttext,min,3-3-4,' \
    f'{best_valid_acc:.3f},' \
    f'{test_acc:.3f},' \
    f'{",".join([f"{r[i] / np.sum(r):.3f}" for i, r in enumerate(cm)])}\n'
with open('../test/results_tobacco.csv', 'a+') as f:
    f.write(s)

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(cm, cmap='hot', interpolation='nearest')
[ax.text(j, i, round(cm[i][j]/np.sum(cm[i]), 2), ha="center", va="center") for i in range(len(cm)) for j in range(len(cm[i]))]
plt.show()

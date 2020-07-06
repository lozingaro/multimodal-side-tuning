"""
Creative Common 4.0 License for reuse with citation
&copy; 2020 Stefano Pio Zingaro
"""

from __future__ import division, print_function

import random
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.rvl_cdip import RvlDataset
from models import TrainingPipeline, FusionSideNetFc

filterwarnings("ignore")
cudnn.deterministic = True
cudnn.benchmark = False
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

num_classes = 16
num_epochs = 10
result_file = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/test/results_rvl.csv'
cm_file = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/test/confusion_matrices/mobilenet_1024_rvl.png'
side_fc = 1024

d_train = RvlDataset(f'{conf.rlv_img_root_dir}/train', f'{conf.rlv_txt_root_dir}/train')
dl_train = DataLoader(d_train, batch_size=48, shuffle=True)
d_val = RvlDataset(f'{conf.rlv_img_root_dir}/val', f'{conf.rlv_txt_root_dir}/val')
dl_val = DataLoader(d_val, batch_size=48, shuffle=True)
d_test = RvlDataset(f'{conf.rlv_img_root_dir}/test', f'{conf.rlv_txt_root_dir}/test')
dl_test = DataLoader(d_test, batch_size=48, shuffle=False)
train_targets = d_train.targets
labels = d_train.classes

model = FusionSideNetFc(300, num_classes=num_classes, alphas=[.2, .3, .5], dropout_prob=.5, side_fc=side_fc).to(conf.core.device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
criterion = nn.CrossEntropyLoss().to(conf.device)
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .1 * (1.0 - float(epoch) / float(num_epochs)) ** .5)
pipeline = TrainingPipeline(model, criterion, optimizer, scheduler, device=conf.device, num_classes=num_classes)
best_valid_acc, test_acc, cm, dist = pipeline.run(dl_train, dl_val, dl_test, num_epochs=num_epochs, classes=labels)

s = f'1280x{side_fc}x10_mobilenet,sgd,fasttext,no,2-3-5,' \
    f'{best_valid_acc:.3f},' \
    f'{test_acc:.3f},' \
    f'{",".join([f"{r[i] / np.sum(r):.3f}" for i, r in enumerate(cm)])}\n'
with open(result_file, 'a+') as f:
    f.write(s)

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(cm, aspect='auto', cmap=plt.get_cmap('Reds'))
plt.ylabel('Actual Category')
plt.yticks(range(len(cm)), labels, rotation=45)
plt.xlabel('Predicted Category')
plt.xticks(range(len(cm)), labels, rotation=45)
[ax.text(j, i, round(cm[i][j]/np.sum(cm[i]), 2), ha="center", va="center") for i in range(len(cm)) for j in range(len(cm[i]))]
fig.tight_layout()
plt.savefig(cm_file)

"""
    Multimodal side-tuning for document classification
    Copyright (C) 2020  S.P. Zingaro <mailto:stefanopio.zingaro@unibo.it>.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

import config
from datasets.rvl_cdip import RvlDataset
from models import TrainingPipeline, FusionSideNetFc

print("""
    Multimodal side-tuning for document classification
    Copyright (C) 2020  Stefano Pio Zingaro <mailto:stefanopio.zingaro@unibo.it>

    This program comes with ABSOLUTELY NO WARRANTY.
    This is free software, and you are welcome to redistribute it
    under certain conditions; visit <http://www.gnu.org/licenses/> for details.
""")

filterwarnings("ignore")
cudnn.deterministic = True
cudnn.benchmark = False
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

d_train = RvlDataset(f'{config.rlv_img_root_dir}/train', f'{config.rlv_txt_root_dir}/train')
dl_train = DataLoader(d_train, batch_size=48, shuffle=True)
d_val = RvlDataset(f'{config.rlv_img_root_dir}/val', f'{config.rlv_txt_root_dir}/val')
dl_val = DataLoader(d_val, batch_size=48, shuffle=True)
d_test = RvlDataset(f'{config.rlv_img_root_dir}/test', f'{config.rlv_txt_root_dir}/test')
dl_test = DataLoader(d_test, batch_size=48, shuffle=False)
train_targets = d_train.targets
labels = d_train.classes

num_classes = len(np.unique(train_targets))
num_epochs = 200
side_fc = 256
alphas = [.3, .3, .4]

model = FusionSideNetFc(300, num_classes=num_classes, alphas=alphas, dropout_prob=.5, side_fc=side_fc).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                              lambda epoch: .1 * (1.0 - float(epoch) / float(num_epochs)) ** .5)
pipeline = TrainingPipeline(model, criterion, optimizer, scheduler, device=device, num_classes=num_classes)
best_valid_acc, test_acc, cm, dist = pipeline.run(dl_train, dl_val, dl_test, num_epochs=num_epochs, classes=labels)

result_file = '../test/results_rvl.csv'
with open(result_file, 'a+') as f:
    f.write(f'1280x{side_fc}x10,'
            f'{sum(p.numel() for p in model.parameters() if p.requires_grad)},'
            f'sgd,'
            f'fasttext,'
            f'min,'
            f'{alphas},'
            f'{best_valid_acc:.3f},'
            f'{test_acc:.3f},'
            f'{",".join([f"{r[i] / np.sum(r):.3f}" for i, r in enumerate(cm)])}\n')

cm_file = f'../test/confusion_matrices/side_{side_fc}_rvl.png'
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(cm, aspect='auto', cmap=plt.get_cmap('Reds'))
plt.ylabel('Actual Category')
plt.yticks(range(len(cm)), labels, rotation=45)
plt.xlabel('Predicted Category')
plt.xticks(range(len(cm)), labels, rotation=45)
[ax.text(j, i, round(cm[i][j]/np.sum(cm[i]), 2), ha="center", va="center") for i in range(len(cm)) for j in range(len(cm[i]))]
fig.tight_layout()
plt.savefig(cm_file)


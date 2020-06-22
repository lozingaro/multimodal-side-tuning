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

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets import TobaccoTxtDataset, RvlTxtDataset
from models.nets import ShawnNet
from models.utils import TrainingPipeline

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

num_epochs = 10

# d_train = RvlTxtDataset(f'{conf.rlv_txt_root_dir}/train')
# dl_train = DataLoader(d_train, batch_size=48, shuffle=True)
# d_val = RvlTxtDataset(f'{conf.rlv_txt_root_dir}/val')
# dl_val = DataLoader(d_val, batch_size=48, shuffle=True)
# d_test = RvlTxtDataset(f'{conf.rlv_txt_root_dir}/test')
# dl_test = DataLoader(d_test, batch_size=48, shuffle=False)
# train_targets = d_train.targets
# labels = d_train.classes

d = TobaccoTxtDataset(conf.tobacco_txt_root_dir)
r = torch.utils.data.random_split(d, [800, 200, 2482])
d_train = r[0]
d_val = r[1]
d_test = r[2]
dl_train = DataLoader(d_train, batch_size=16, shuffle=True)
dl_val = DataLoader(d_val, batch_size=4, shuffle=True)
dl_test = DataLoader(d_test, batch_size=32, shuffle=False)
train_targets = d_train.dataset.targets
labels = d.classes

num_classes = np.unique(train_targets)
model = ShawnNet(300, num_filters=512, windows=[3, 4, 5], num_classes=num_classes, dropout_prob=.5).to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
_, c = np.unique(np.array(train_targets), return_counts=True)
weight = torch.from_numpy(np.min(c) / c).float().to(conf.device)
criterion = nn.CrossEntropyLoss(weight=weight).to(conf.device)
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .1 * (1.0 - float(epoch) / float(num_epochs)) ** .5)
pipeline = TrainingPipeline(model, criterion, optimizer, scheduler, device=conf.device, num_classes=num_classes)
best_valid_acc, test_acc, cm, dist = pipeline.run(dl_train, dl_val, dl_test, num_epochs=num_epochs, classes=labels)

result_file = '../test/results_rvl.csv'
with open(result_file, 'a+') as f:
    f.write(f'shawn,sgd,fasttext,min,-,'
            f'{best_valid_acc:.3f},'
            f'{test_acc:.3f},'
            f'{",".join([f"{r[i] / np.sum(r):.3f}" for i, r in enumerate(cm)])}\n')

cm_file = '../test/confusion_matrices/shawn_rvl.png'
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(cm, aspect='auto', cmap=plt.get_cmap('Reds'))
plt.ylabel('Actual Category')
plt.yticks(range(len(cm)), labels, rotation=45)
plt.xlabel('Predicted Category')
plt.xticks(range(len(cm)), labels, rotation=45)
[ax.text(j, i, round(cm[i][j]/np.sum(cm[i]), 2), ha="center", va="center") for i in range(len(cm)) for j in range(len(cm[i]))]
fig.tight_layout()
plt.savefig(cm_file)

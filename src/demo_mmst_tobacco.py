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
import time
from warnings import filterwarnings

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import config
from datasets.tobacco import TobaccoDataset
from models import TrainingPipeline, FusionSideNetFcMobileNet, FusionSideNetFcResNet

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

d = TobaccoDataset(config.tobacco_img_root_dir, config.tobacco_txt_root_dir)
d_train, d_val, d_test = torch.utils.data.random_split(d, [800, 200, 2482])
dl_train = DataLoader(d_train, batch_size=16, shuffle=True)
dl_val = DataLoader(d_val, batch_size=4, shuffle=True)
dl_test = DataLoader(d_test, batch_size=32, shuffle=False)

num_classes = len(d.classes)
train_targets = d_train.dataset.targets
labels = d.classes
num_epochs = 100

for alphas in config.alphas[4:5]:
    print(alphas)
    for model in (
        # FusionSideNetFcMobileNet(300, num_classes=num_classes, alphas=alphas, dropout_prob=.5, side_fc=1024),
        # FusionSideNetFcMobileNet(300, num_classes=num_classes, alphas=alphas, dropout_prob=.5, side_fc=512),
        # FusionSideNetDirect(300, num_classes=num_classes, alphas=alphas, dropout_prob=.5),
        FusionSideNetFcResNet(300, num_classes=num_classes, alphas=alphas, dropout_prob=.5, side_fc=1024),
        # FusionSideNetFcVGG(300, num_classes=num_classes, alphas=alphas, dropout_prob=.5, side_fc=512),
    ):
        model = model.to(device)
        learning_rate = .1
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)
        criterion = nn.CrossEntropyLoss().to(device)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: learning_rate * (1.0 - float(epoch) / num_epochs) ** .5
        )
        pipeline = TrainingPipeline(model, criterion, optimizer, scheduler, device=device, num_classes=num_classes)

        since = time.time()
        best_valid_acc, test_acc, cm, dist = pipeline.run(dl_train, dl_val, dl_test, num_epochs=num_epochs,
                                                          classes=labels)
        time_elapsed = time.time() - since

        result_file = '../test/results_tobacco.csv'
        with open(result_file, 'a+') as f:
            f.write(f'{model.name},'
                    f'{round(time_elapsed)},'
                    f'{sum(p.numel() for p in model.parameters() if p.requires_grad)},'
                    f'sgd,'
                    f'fasttext,'
                    f'no,'
                    f'{"-".join([str(i) for i in alphas])},'
                    f'{best_valid_acc:.3f},'
                    f'{test_acc:.3f},'
                    f'{",".join([f"{r[i] / np.sum(r):.3f}" for i, r in enumerate(cm)])}\n')

        # plot_cm(cm,
        #         labels,
        #         f'../test/confusion_matrices/{model.name}_tobacco_{"-".join([str(i) for i in alphas])}.png')

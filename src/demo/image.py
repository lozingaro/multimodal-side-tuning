from __future__ import division, print_function

from collections import OrderedDict
from warnings import filterwarnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
import random
from datasets.tobacco import ImageDataset
from models.nets import MobileNet
from models.utils import TrainingPipeline

filterwarnings("ignore")
cudnn.deterministic = True
cudnn.benchmark = False
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

dataset = ImageDataset(conf.image_root_dir,
                       image_width=384,
                       image_interpolation=Image.BILINEAR,
                       image_mean_norm=[0.485, 0.456, 0.406],
                       image_std_norm=[0.229, 0.224, 0.225])
random_dataset_split = torch.utils.data.random_split(dataset, [800, 200, 2482])
datasets = OrderedDict({
    'train': random_dataset_split[0],
    'val': random_dataset_split[1],
    'test': random_dataset_split[2]
})
dataloaders = {
    x: DataLoader(datasets[x],
                  batch_size=conf.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'))
    for x in conf.batch_sizes
}
model = MobileNet(num_classes=10, dropout_prob=.5).to(conf.core.device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
_, c = np.unique(np.array(dataset.targets)[dataloaders['train'].dataset.indices], return_counts=True)
weight = torch.from_numpy(np.min(c) / c).type(torch.FloatTensor).to(conf.device)
criterion = nn.CrossEntropyLoss(weight=weight).to(conf.device)
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .1 * (1.0 - float(epoch) / 100.0) ** .5)
pipeline = TrainingPipeline(model, criterion, optimizer, scheduler, device=conf.device)
best_valid_acc, test_acc, confusion_matrix = pipeline.run(dataloaders['train'],
                                                          dataloaders['val'],
                                                          dataloaders['test'],
                                                          num_epochs=100)
s = f'mobilenetv2,sgd,-,no,-,' \
    f'{best_valid_acc:.3f},' \
    f'{test_acc:.3f},' \
    f'{",".join([f"{r[i] / np.sum(r):.3f}" for i, r in enumerate(confusion_matrix)])}\n'
with open('../test/results.csv', 'a+') as f:
    f.write(s)

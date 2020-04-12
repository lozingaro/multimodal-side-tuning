from __future__ import division, print_function

from collections import OrderedDict
from warnings import filterwarnings

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
import random
import fasttext
from datasets.tobacco import TextDataset
from models import TrainingPipeline
from models.nets import ShawnNet

filterwarnings("ignore")
cudnn.deterministic = True
cudnn.benchmark = False
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

nlp = fasttext.load_model(conf.core.text_fasttext_model_path)
dataset = TextDataset(conf.text_root_dir, nlp=nlp)
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
model = ShawnNet(300,
                 windows=[3, 4, 5],
                 dropout_prob=.5).to(conf.core.device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
_, c = np.unique(np.array(dataset.targets)[dataloaders['train'].dataset.indices], return_counts=True)
weight = torch.from_numpy(np.min(c) / c).type(torch.FloatTensor).to(conf.device)
criterion = nn.CrossEntropyLoss(weight=weight).to(conf.device)
optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
pipeline = TrainingPipeline(model, criterion, optimizer, device=conf.core.device)
best_valid_acc, test_acc, confusion_matrix = pipeline.run(dataloaders['train'],
                                                          dataloaders['val'],
                                                          dataloaders['test'],
                                                          num_epochs=100)
s = f'shawn,sgd,fasttext,no,-,' \
    f'{best_valid_acc:.3f},' \
    f'{test_acc:.3f},' \
    f'{",".join([f"{r[i] / np.sum(r):.3f}" for i, r in enumerate(confusion_matrix)])}\n'
with open('../test/results.csv', 'a+') as f:
    f.write(s)

from __future__ import print_function, division

import math
from os import path
from warnings import filterwarnings

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import datasets
import models
from conf import core, model

filterwarnings("ignore")

# Set the seed for pseudorandom operations
torch.manual_seed(core.seed)
cudnn.deterministic = True

print('\nLoading data...')
image_dataset_saved_path = '/tmp/tobacco_image_dataset_224_bilinear_norm.pth'
if path.exists(image_dataset_saved_path) and core.load_dataset:
    image_dataset = torch.load(image_dataset_saved_path)
else:
    image_dataset = datasets.tobacco.TobaccoImageDataset(core.root_dir, list(core.lengths.values()))
    torch.save(image_dataset, image_dataset_saved_path)

image_dataloaders = {
    x: DataLoader(image_dataset.datasets[x],
                  batch_size=core.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'),
                  num_workers=0,
                  pin_memory=True)
    for x in ['train', 'val', 'test']
}

print('\nTraining model...')
image_model = models.resnet.SideTuneModel(len(image_dataset.classes), alpha=model.image_alpha)
# image_model = models.mobilenet.SideTuneModel(len(image_dataset.classes), alpha=model.image_alpha)

image_criterion = nn.CrossEntropyLoss()
image_optimizer = torch.optim.SGD(image_model.parameters(), lr=model.image_initial_lr, momentum=model.image_momentum)
image_scheduler = LambdaLR(image_optimizer,
                           lambda epoch: model.image_initial_lr * math.sqrt(1 - epoch / model.image_max_epochs))
image_model = models.utils.train_eval_test(image_model,
                                           image_dataloaders,
                                           criterion=image_criterion,
                                           optimizer=image_optimizer,
                                           scheduler=image_scheduler,
                                           lengths=core.lengths,
                                           num_epochs=model.image_max_epochs)

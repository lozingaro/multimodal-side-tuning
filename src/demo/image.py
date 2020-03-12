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
import conf

filterwarnings("ignore")

# Set the seed for pseudorandom operations
torch.manual_seed(conf.core.seed)
cudnn.deterministic = True

print('\nLoading data...')
image_dataset_saved_path = '/tmp/tobacco_image_dataset_224_bilinear_norm.pth'
if path.exists(image_dataset_saved_path) and conf.core.load_image_dataset:
    image_dataset = torch.load(image_dataset_saved_path)
else:
    image_dataset = datasets.tobacco.TobaccoImageDataset(conf.dataset.image_root_dir, list(conf.dataset.image_len.values()))
    torch.save(image_dataset, image_dataset_saved_path)

image_dataloaders = {
    x: DataLoader(image_dataset.datasets[x],
                  batch_size=conf.dataset.image_batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'),
                  num_workers=0,
                  pin_memory=True)
    for x in ['train', 'val', 'test']
}

print('\nTraining model...')

# image_model = models.resnet.SideTuneModel(len(image_dataset.classes))
image_model = models.mobilenet.FineTuneModel(len(image_dataset.classes))

image_criterion = nn.CrossEntropyLoss()
image_optimizer = torch.optim.SGD(image_model.parameters(), conf.model.image_initial_lr, conf.model.image_momentum)
image_scheduler = LambdaLR(image_optimizer,
                           lambda epoch: conf.model.image_initial_lr * math.sqrt(1 - epoch / conf.model.image_num_epochs))
image_training_pipeline = models.utils.TrainingPipeline(train_function=models.utils.image_train,
                                                        eval_function=models.utils.image_eval,
                                                        test_function=models.utils.image_test)
image_model = image_training_pipeline.train_eval_test(image_model,
                                                      image_dataloaders,
                                                      image_optimizer,
                                                      image_criterion,
                                                      image_scheduler,
                                                      conf.dataset.image_len,
                                                      conf.model.image_num_epochs)

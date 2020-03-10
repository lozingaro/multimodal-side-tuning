from __future__ import print_function, division

from math import sqrt
from os import path
from warnings import filterwarnings

import torch
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
# image_dataset_saved_path = '/tmp/tobacco_image_dataset_bicubic_norm.pth'
image_dataset_saved_path = '/tmp/tobacco_image_dataset_bilinear_norm.pth'
# image_dataset_saved_path = '/tmp/tobacco_image_dataset_pad_bilinear_norm.pth'
# image_dataset_saved_path = '/tmp/tobacco_image_dataset_pad_rotate_bilinear_norm.pth'
if path.exists(image_dataset_saved_path) and core.load_dataset:
    image_dataset = torch.load(image_dataset_saved_path)
else:
    image_dataset = datasets.tobacco.TobaccoImageDataset(core.root_dir, list(core.lengths.values()))
    torch.save(image_dataset, image_dataset_saved_path)

image_dataloaders = {
    x: DataLoader(image_dataset.datasets[x],
                  batch_size=core.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'),
                  num_workers=core.workers)
    for x in ['train', 'val', 'test']
}

print('\nLoading model...')
image_model = models.additive.FineTuneModel(len(image_dataset.classes))
image_model = image_model.to(core.device)
image_model_saved_path = '/tmp/tobacco_image_model_bilinear_norm_01.pth'
if path.exists(image_model_saved_path) and core.load_dataset:
    image_model = torch.load(image_model_saved_path)
else:
    image_optimizer = torch.optim.SGD(image_model.parameters(), lr=model.image_initial_lr,
                                      momentum=model.image_momentum)
    lr_lambda = lambda epoch: model.image_initial_lr * sqrt(1 - epoch / model.image_epochs)
    image_scheduler = LambdaLR(image_optimizer, lr_lambda=lr_lambda)
    image_model = models.utils.train_image_model(image_model,
                                                 image_dataloaders,
                                                 model.image_criterion,
                                                 optimizer=image_optimizer,
                                                 scheduler=image_scheduler,
                                                 device=core.device,
                                                 lengths=core.lengths,
                                                 num_epochs=model.image_epochs)
    torch.save(image_model, image_model_saved_path)

print('\nTesting on the remaining samples...')
acc = models.evaluate_model(image_model, image_dataloaders['test'], device=core.device, length=core.lengths['test'])
print('Test Accuracy: {:4f}'.format(acc))

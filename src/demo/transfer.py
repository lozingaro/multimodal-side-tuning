from __future__ import print_function, division

from os import path
from warnings import filterwarnings

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import datasets
import models
from conf import core, model

filterwarnings("ignore")

# Set the seed for pseudorandom operations
torch.manual_seed(core.seed)
cudnn.deterministic = True

print('\nLoading data...')
image_dataset_saved_path = '/tmp/tobacco_image_dataset_bilinear_norm.pth'
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

print('\nTraining model...')
image_model = models.resnet.FineTuneModel(len(image_dataset.classes), freeze=True)

image_criterion = nn.CrossEntropyLoss()

if core.use_gpu:
    image_model.cuda()
    image_criterion.cuda()

image_optimizer = torch.optim.SGD(image_model.parameters(), lr=model.image_initial_lr, momentum=model.image_momentum)

image_model = models.utils.train_image_model(image_model,
                                             image_dataloaders,
                                             criterion=image_criterion,
                                             optimizer=image_optimizer,
                                             scheduler=models.utils.custom_scheduler(image_optimizer),
                                             lengths=core.lengths,
                                             num_epochs=model.image_epochs)

# Save for further use
image_model.save_state_dict('/tmp/tobacco_image_model_bilinear_norm_01.pth')

print('\nTesting on the remaining samples...')
acc = models.evaluate_model(image_model, image_dataloaders['test'], length=core.lengths['test'])
print('Test Accuracy: {:4f}'.format(acc))


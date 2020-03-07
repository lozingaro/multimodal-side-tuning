from __future__ import print_function, division

import os
import warnings

import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from configs import core
from my_datasets.tobacco import TobaccoImageDataset
from my_datasets.utils import imshow
from my_models.transfer import FineTuneModel
from my_models.utils import visualize_model, train_model

warnings.filterwarnings("ignore")

# Load Data
image_transforms = {
    # Data augmentation and normalization for training
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(core.mean_normalization, core.std_normalization),
    ]),
    # Just normalization for validation
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(core.mean_normalization, core.std_normalization),
    ]),
    # Just normalization for validation, check on the significance of Resize(224)
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(core.mean_normalization, core.std_normalization),
    ]),
}

image_dataset_saved_path = '/tmp/tobacco_image_dataset.pth'
if os.path.exists(image_dataset_saved_path) and not core.build_dataset_from_scratch:
    image_dataset = torch.load(image_dataset_saved_path)
else:
    image_dataset = TobaccoImageDataset(core.root_dir, list(core.lengths.values()), image_transforms)
    torch.save(image_dataset, image_dataset_saved_path)

image_dataloaders = {
    x: DataLoader(image_dataset.datasets[x],
                  batch_size=core.batch_sizes[x],
                  shuffle=x == 'train',
                  num_workers=core.workers,
                  pin_memory=True)
    for x in ['train', 'val', 'test']
}

# Visualize a few images
one_batch_inputs, one_batch_classes = next(iter(image_dataloaders['train']))
out = utils.make_grid(one_batch_inputs)
imshow(out, title=[image_dataset.classes[x] for x in one_batch_classes])

# Finetuning the convnet
image_model = FineTuneModel(len(image_dataset.classes))
image_model = image_model.to(core.device)

train_criterion = nn.CrossEntropyLoss()
optimizer_ft = Adam(image_model.parameters(), lr=0.001, weight_decay=0.0)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # maybe useless?

# Train and evaluate
model_ft = train_model(image_dataloaders,
                       image_model,
                       train_criterion,
                       optimizer_ft,
                       exp_lr_scheduler,
                       device=core.device,
                       lengths=core.lengths,
                       num_epochs=1)

visualize_model(model_ft, dataloader=image_dataloaders['test'], device=core.device)

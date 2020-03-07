from __future__ import print_function, division

import os
import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms, utils

from conf import core, model
from datasets.tobacco import TobaccoImageDataset
import datasets.utils
from models.transfer import FineTuneModel
from models.utils import evaluate_model, train_image_model

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
datasets.utils.imshow(out, title=[image_dataset.classes[x] for x in one_batch_classes])

# Finetuning the convnet
image_model = FineTuneModel(len(image_dataset.classes))
image_model = image_model.to(core.device)
image_train_criterion = nn.CrossEntropyLoss()

# Train and evaluate
optimizer = model.adam_optimizer(image_model)
scheduler = model.expr_lr_scheduler(optimizer)
image_model = train_image_model(image_dataloaders,
                                image_model,
                                image_train_criterion,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                device=core.device,
                                lengths=core.lengths,
                                num_epochs=core.epochs)

evaluate_model(image_model, dataloader=image_dataloaders['test'], device=core.device)
plt.show()

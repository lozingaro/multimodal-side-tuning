from __future__ import print_function, division

from math import pow
from os import path
from warnings import filterwarnings

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

import datasets
import models
from conf import core, model

filterwarnings("ignore")

# Load Data
image_transforms = {
    # Data augmentation and normalization for training
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(core.mean_normalization, core.std_normalization),
    ]),
    # Just normalization for validation
    'val': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(core.mean_normalization, core.std_normalization),
    ]),
    # Just normalization for validation
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(core.mean_normalization, core.std_normalization),
    ]),
}

image_dataset_saved_path = '/tmp/tobacco_image_dataset.pth'
if path.exists(image_dataset_saved_path) and not core.build_dataset_from_scratch:
    image_dataset = torch.load(image_dataset_saved_path)
else:
    image_dataset = datasets.tobacco.TobaccoImageDataset(core.root_dir, list(core.lengths.values()),
                                                         image_transforms)
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
# one_batch_inputs, one_batch_classes = next(iter(image_dataloaders['train']))
# out = utils.make_grid(one_batch_inputs)
# datasets.utils.imshow(out, title=[image_dataset.classes[x] for x in one_batch_classes])

# Finetuning the convnet
image_model = models.transfer.FineTuneModel(len(image_dataset.classes))
image_model = image_model.to(core.device)

# Train and evaluate
image_model_saved_path = '/tmp/tobacco_image_model.pth'
if path.exists(image_model_saved_path) and not core.build_model_from_scratch:
    image_model = torch.load(image_model_saved_path)
else:
    image_optimizer = torch.optim.SGD(image_model.parameters(), lr=model.image_initial_lr,
                                      momentum=model.image_momentum)
    image_scheduler = LambdaLR(image_optimizer, lr_lambda=lambda epoch: pow(1 - epoch / model.image_epochs, .5))
    image_model = models.utils.train_image_model(image_model,
                                                 image_dataloaders,
                                                 model.image_criterion,
                                                 optimizer=image_optimizer,
                                                 scheduler=image_scheduler,
                                                 device=core.device,
                                                 lengths=core.lengths,
                                                 num_epochs=model.image_epochs)
    torch.save(image_model, image_model_saved_path)

test_accuracy = models.utils.evaluate_model(image_model,
                                            image_dataloaders['test'],
                                            length=core.lengths['test'],
                                            device=core.device)

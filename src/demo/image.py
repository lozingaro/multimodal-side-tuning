from __future__ import print_function, division

import math
from warnings import filterwarnings

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.tobacco import TobaccoImageDataset
from models.nets import MobileNet, ResNet
from models.utils import TrainingPipeline

filterwarnings("ignore")

torch.manual_seed(conf.core.seed)
cudnn.deterministic = True

print('\nLoading data...', end=' ')
dataset = TobaccoImageDataset(conf.dataset.image_root_dir,
                              image_width=conf.dataset.image_width,
                              image_interpolation=conf.dataset.image_interpolation,
                              image_mean_norm=conf.dataset.image_mean_normalization,
                              image_std_norm=conf.dataset.image_std_normalization,
                              splits=conf.dataset.lengths)
dataloaders = {
    x: DataLoader(dataset.datasets[x],
                  batch_size=conf.dataset.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'),
                  num_workers=0,
                  pin_memory=True)
    for x in conf.dataset.batch_sizes
}
print('done.')

print('\nModel train and evaluation... parameters=', end='')
model = MobileNet(len(dataset.classes), alpha=conf.alpha).to(conf.core.device)
print(sum([p.numel() for p in model.parameters()]))
criterion = torch.nn.CrossEntropyLoss().to(conf.core.device)
# image_optimizer = torch.optim.Adam(image_model.parameters(), lr=conf.model.image_lr)
optimizer = torch.optim.SGD(model.parameters(), lr=conf.model.image_lr, momentum=conf.model.momentum)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                              lambda epoch: conf.model.image_lr * math.sqrt(
                                                        1 - epoch / conf.model.image_num_epochs))
pipeline = TrainingPipeline(model,
                            optimizer,
                            criterion,
                            scheduler,
                            device=conf.core.device)
pipeline.run(dataloaders['train'],
             dataloaders['val'],
             dataloaders['test'],
             num_epochs=conf.model.image_num_epochs)

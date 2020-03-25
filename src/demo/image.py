from __future__ import division, print_function

from warnings import filterwarnings

import torch
import torch.nn as nn
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.tobacco import ImageDataset
from models.nets import MobileNet
from models.utils import TrainingPipeline

filterwarnings("ignore")
torch.manual_seed(101)
cudnn.deterministic = True

print('\nLoading data...', end=' ')
dataset = ImageDataset(conf.core.image_root_dir,
                       image_width=384,
                       image_interpolation=Image.BILINEAR,
                       image_mean_norm=[0.485, 0.456, 0.406],
                       image_std_norm=[0.229, 0.224, 0.225])
dataloaders = {
    x: DataLoader(dataset.datasets[x],
                  batch_size=conf.core.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'))
    for x in conf.core.batch_sizes
}
print('done.')

model = MobileNet(len(dataset.classes), alpha=.55).to(conf.core.device)
print(f'\nModel train (model parameters={sum([p.numel() for p in model.parameters() if p.requires_grad])})...')
criterion = nn.CrossEntropyLoss().to(conf.core.device)
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                              lambda epoch: .1 * (1 - epoch / 100)**.5)
pipeline = TrainingPipeline(model, criterion, optimizer, scheduler, device=conf.core.device)
pipeline.run(dataloaders['train'], dataloaders['val'], dataloaders['test'], num_epochs=100)

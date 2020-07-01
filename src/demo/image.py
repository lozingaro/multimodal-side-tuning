from __future__ import division, print_function

from collections import OrderedDict
from warnings import filterwarnings

import numpy as np
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
torch.manual_seed(42)
cudnn.deterministic = True
try:
    dataset = torch.load(conf.image_dataset_path)
except FileNotFoundError:
    dataset = ImageDataset(conf.core.image_root_dir,
                           image_width=384,
                           image_interpolation=Image.BILINEAR,
                           image_mean_norm=[0.485, 0.456, 0.406],
                           image_std_norm=[0.229, 0.224, 0.225])
    torch.save(dataset, conf.image_dataset_path)
random_dataset_split = torch.utils.data.random_split(dataset, [800, 200, 2482])
datasets = OrderedDict({
    'train': random_dataset_split[0],
    'val': random_dataset_split[1],
    'test': random_dataset_split[2]
})
dataloaders = {
    x: DataLoader(datasets[x],
                  batch_size=conf.core.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'))
    for x in conf.core.batch_sizes
}
model = MobileNet(len(dataset.classes), alpha=.5).to(conf.core.device)
_, c = np.unique(np.array(dataset.targets)[dataloaders['train'].dataset.indices], return_counts=True)
weights = torch.from_numpy(np.min(c) / c).type(torch.FloatTensor).to(conf.core.device)
criterion = nn.CrossEntropyLoss(weight=weights).to(conf.core.device)
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .1 * (1 - epoch / 100) ** .5)
pipeline = TrainingPipeline(model, criterion, optimizer, scheduler, device=conf.core.device)
pipeline.run(dataloaders['train'], dataloaders['val'], dataloaders['test'], num_epochs=100)

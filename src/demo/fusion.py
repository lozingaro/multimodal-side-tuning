from __future__ import division, print_function

from warnings import filterwarnings

import numpy as np
import fasttext
import torch
import torch.nn as nn
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.tobacco import FusionDataset, ImageDataset, TextDataset
from models import TextImageSideNet, TrainingPipeline, TextImageSideNetBaseFC

filterwarnings("ignore")
torch.manual_seed(42)
cudnn.deterministic = True

print('\nLoading data...', end=' ')
image_dataset = ImageDataset(conf.core.image_root_dir,
                             image_width=384,
                             image_interpolation=Image.BILINEAR,
                             image_mean_norm=[0.485, 0.456, 0.406],
                             image_std_norm=[0.229, 0.224, 0.225])
nlp = fasttext.load_model(conf.core.text_fasttext_model_path)
text_dataset = TextDataset(conf.core.text_root_dir, nlp=nlp)
dataset = FusionDataset(image_dataset, text_dataset)
dataloaders = {
    x: DataLoader(dataset.datasets[x],
                  batch_size=conf.core.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'))
    for x in conf.core.batch_sizes
}
print('done.')

model = TextImageSideNet(300,
                         num_classes=10,
                         alphas=[.4, .4],
                         dropout_prob=.5).to(conf.core.device)
print(
    f'\nModel train (trainable model parameters={sum([p.numel() for p in model.parameters() if p.requires_grad])})...')
_, c = np.unique(np.array(dataset.targets)[dataloaders['train'].dataset.indices], return_counts=True)
weights = torch.from_numpy(np.min(c) / c).type(torch.FloatTensor).to(conf.core.device)
criterion = nn.CrossEntropyLoss(weight=weights).to(conf.core.device)
optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
#                                               lambda epoch: .1 * (1 - epoch / 100) ** .5)
pipeline = TrainingPipeline(model, criterion, optimizer, device=conf.core.device)
pipeline.run(dataloaders['train'], dataloaders['val'], dataloaders['test'], num_epochs=100)

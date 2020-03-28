from __future__ import division, print_function

from warnings import filterwarnings

import numpy as np
import fasttext
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.tobacco import TextDataset
from models import TrainingPipeline
from models.nets import ShawnNet

filterwarnings("ignore")
torch.manual_seed(42)
cudnn.deterministic = True

print('\nLoading data...', end=' ')
# nlp = fasttext.load_model(conf.core.text_fasttext_model_path)
dataset = TextDataset(conf.core.text_root_dir, nlp=None)
dataloaders = {
    x: DataLoader(dataset.datasets[x],
                  batch_size=conf.core.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'))
    for x in conf.core.batch_sizes
}
print('done.')

model = ShawnNet(300,
                 windows=[3, 4, 5],
                 dropout_prob=.5,
                 custom_embedding=True,
                 custom_num_embeddings=len(dataset.lookup)).to(conf.core.device)
print(f'\nModel train (trainable model parameters={sum([p.numel() for p in model.parameters() if p.requires_grad])})...')
_, c = np.unique(np.array(dataset.targets)[dataloaders['train'].dataset.indices], return_counts=True)
weights = torch.from_numpy(np.min(c) / c).type(torch.FloatTensor).to(conf.core.device)
criterion = nn.CrossEntropyLoss(weight=weights).to(conf.core.device)
optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
pipeline = TrainingPipeline(model, criterion, optimizer, device=conf.core.device)
pipeline.run(dataloaders['train'], dataloaders['val'], dataloaders['test'], num_epochs=100)

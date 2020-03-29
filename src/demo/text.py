from __future__ import division, print_function

from collections import OrderedDict
from warnings import filterwarnings

import numpy as np
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
try:
    dataset = torch.load(conf.text_dataset_custom_path)
except FileNotFoundError:
    nlp = None  # fasttext.load_model(conf.core.text_fasttext_model_path)
    dataset = TextDataset(conf.core.text_root_dir, nlp=nlp)
    torch.save(dataset, conf.text_dataset_custom_path)
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
model = ShawnNet(300,
                 windows=[3, 4, 5],
                 dropout_prob=.5,
                 custom_embedding=True,
                 custom_num_embeddings=len(dataset.lookup)).to(conf.core.device)
_, c = np.unique(np.array(dataset.targets)[dataloaders['train'].dataset.indices], return_counts=True)
weights = torch.from_numpy(np.min(c) / c).type(torch.FloatTensor).to(conf.core.device)
criterion = nn.CrossEntropyLoss(weight=weights).to(conf.core.device)
optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
pipeline = TrainingPipeline(model, criterion, optimizer, device=conf.core.device)
pipeline.run(dataloaders['train'], dataloaders['val'], dataloaders['test'], num_epochs=100)

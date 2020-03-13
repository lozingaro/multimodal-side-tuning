from __future__ import print_function, division

from warnings import filterwarnings

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.tobacco import TobaccoTextDataset
from datasets.utils import generate_text_batch
from models import TextTrainingPipeline
from models.cnn import TextSentiment

filterwarnings("ignore")

torch.manual_seed(conf.core.seed)
cudnn.deterministic = True

print('\nLoading data...', end=' ')
text_dataset = TobaccoTextDataset(conf.dataset.text_root_dir, conf.dataset.text_lengths)
text_dataloaders = {
    x: DataLoader(text_dataset.datasets[x],
                  batch_size=conf.dataset.text_batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'),
                  collate_fn=generate_text_batch,
                  num_workers=0,
                  pin_memory=True)
    for x in conf.dataset.text_batch_sizes
}
print('done.')

print('\nModel train and evaluation...')
text_model = TextSentiment(len(text_dataset.get_vocab()),
                           conf.dataset.text_embedding_dim,
                           len(text_dataset.get_labels())).to(conf.core.device)
text_criterion = torch.nn.CrossEntropyLoss().to(conf.core.device)
text_optimizer = torch.optim.SGD(text_model.parameters(), lr=conf.model.text_lr)
text_scheduler = torch.optim.lr_scheduler.StepLR(text_optimizer, step_size=1, gamma=.9)

pipeline = TextTrainingPipeline(text_model,
                                text_optimizer,
                                text_criterion,
                                text_scheduler)
pipeline.run(text_dataloaders['train'],
             text_dataloaders['val'],
             text_dataloaders['test'],
             conf.model.text_num_epochs)

from __future__ import print_function, division

from warnings import filterwarnings

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.tobacco import TobaccoTextDataset
from models import TrainingPipeline
from models.nets import ShawnNet, CedricNet

filterwarnings("ignore")

torch.manual_seed(conf.core.seed)
cudnn.deterministic = True

print('\nLoading data...', end=' ')
text_dataset = TobaccoTextDataset(conf.dataset.text_root_dir,
                                  context=conf.dataset.text_words_per_doc,
                                  splits=conf.dataset.lengths,
                                  # nlp_model_path=conf.dataset.text_spacy_model_path,
                                  )
text_dataloaders = {
    x: DataLoader(text_dataset.datasets[x],
                  batch_size=conf.dataset.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'),
                  num_workers=0,
                  pin_memory=True)
    for x in conf.dataset.batch_sizes
}
print('done.')

print('\nModel train and evaluation... parameters=', end='')
text_model = ShawnNet(len(text_dataset.vocab),
                      conf.dataset.text_embedding_dim,
                      num_classes=len(text_dataset.classes)).to(conf.core.device)
print(sum([p.numel() for p in text_model.parameters()]))
text_criterion = torch.nn.CrossEntropyLoss().to(conf.core.device)
# text_optimizer = torch.optim.SGD(text_model.parameters(), lr=conf.model.text_lr, momentum=conf.model.text_lr)
text_optimizer = torch.optim.Adam(text_model.parameters(), lr=conf.model.text_lr)
pipeline = TrainingPipeline(text_model,
                            text_optimizer,
                            text_criterion,
                            device=conf.core.device)
pipeline.run(text_dataloaders['train'],
             text_dataloaders['val'],
             text_dataloaders['test'],
             num_epochs=conf.model.text_num_epochs)

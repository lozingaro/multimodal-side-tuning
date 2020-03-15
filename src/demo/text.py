from __future__ import print_function, division

from warnings import filterwarnings

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.tobacco import TobaccoTextDataset
from models import TrainingPipeline
from models.nets import TextClassificationModel

filterwarnings("ignore")

torch.manual_seed(conf.core.seed)
cudnn.deterministic = True

print('\nLoading data...', end=' ')
text_dataset = TobaccoTextDataset(conf.dataset.text_root_dir,
                                  context=conf.dataset.text_words_per_doc,
                                  num_grams=conf.dataset.text_ngrams,
                                  splits=conf.dataset.lengths,
                                  fasttext_model_path=conf.dataset.text_fasttext_model_path)
text_dataloaders = {
    x: DataLoader(text_dataset.datasets[x],
                  batch_size=conf.dataset.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'),
                  num_workers=0,
                  pin_memory=True)
    for x in conf.dataset.batch_sizes
}
print('done.')

print('\nModel train and evaluation...')
text_model = TextClassificationModel(len(text_dataset.vocab),
                                     conf.dataset.text_embedding_dim,
                                     num_classes=len(text_dataset.classes)).to(conf.core.device)
text_criterion = torch.nn.CrossEntropyLoss().to(conf.core.device)
text_optimizer = torch.optim.SGD(text_model.parameters(), lr=conf.model.text_lr, momentum=conf.model.text_lr)

pipeline = TrainingPipeline(text_model,
                            text_optimizer,
                            text_criterion,
                            device=conf.core.device)
pipeline.run(text_dataloaders['train'],
             text_dataloaders['val'],
             text_dataloaders['test'],
             num_epochs=conf.model.text_num_epochs)

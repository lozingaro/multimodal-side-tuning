from __future__ import print_function, division

from warnings import filterwarnings

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.tobacco import TobaccoImageDataset, TobaccoTextDataset
from models.nets import AgneseNetModel

filterwarnings("ignore")

torch.manual_seed(conf.core.seed)
cudnn.deterministic = True

print('\nLoading data...', end=' ')
image_dataset = TobaccoImageDataset(conf.dataset.image_root_dir,
                                    image_width=conf.dataset.image_width,
                                    image_interpolation=conf.dataset.image_interpolation,
                                    image_mean_norm=conf.dataset.image_mean_normalization,
                                    image_std_norm=conf.dataset.image_std_normalization,
                                    splits=conf.dataset.lengths)
image_dataloaders = {
    x: DataLoader(image_dataset.datasets[x],
                  batch_size=conf.dataset.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'),
                  num_workers=0,
                  pin_memory=True)
    for x in conf.dataset.batch_sizes
}
text_dataset = TobaccoTextDataset(conf.dataset.text_root_dir,
                                  context=conf.dataset.text_words_per_doc,
                                  num_grams=conf.dataset.text_ngrams,
                                  splits=conf.dataset.lengths)
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
fusion_model = AgneseNetModel().to(conf.core.device)
fusion_criterion = torch.nn.CrossEntropyLoss().to(conf.core.device)
fusion_optimizer = torch.optim.SGD(fusion_model.parameters(), lr=conf.model.fusion_lr, momentum=conf.model.momentum)

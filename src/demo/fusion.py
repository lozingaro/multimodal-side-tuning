from __future__ import print_function, division

from warnings import filterwarnings

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.tobacco import TobaccoImageDataset, TobaccoTextDataset, TobaccoFusionDataset
from models import FusionTrainingPipeline
from models.nets import TextSideNet_baseFC, TextSideNet, TextSideNet_ResNet

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
text_dataset = TobaccoTextDataset(conf.dataset.text_root_dir,
                                  context=conf.dataset.text_words_per_doc,
                                  splits=conf.dataset.lengths)
fusion_dataset = TobaccoFusionDataset(image_dataset, text_dataset, splits=conf.dataset.lengths)
fusion_dataloaders = {
    x: DataLoader(fusion_dataset.datasets[x],
                  batch_size=conf.dataset.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'),
                  num_workers=0,
                  pin_memory=True)
    for x in conf.dataset.batch_sizes
}
print('done.')

print('\nModel train and evaluation... parameters=', end='')
fusion_model = TextSideNet_baseFC(len(text_dataset.vocab),
                                  conf.dataset.text_embedding_dim,
                                  num_classes=len(text_dataset.classes),
                                  alpha=conf.core.alpha).to(conf.core.device)
print(sum([p.numel() for p in fusion_model.parameters()]))
fusion_criterion = torch.nn.CrossEntropyLoss().to(conf.core.device)
fusion_optimizer = torch.optim.Adam(fusion_model.parameters(), lr=conf.model.fusion_lr)
pipeline = FusionTrainingPipeline(fusion_model,
                                  fusion_optimizer,
                                  fusion_criterion,
                                  device=conf.core.device)
pipeline.run(fusion_dataloaders['train'],
             fusion_dataloaders['val'],
             fusion_dataloaders['test'],
             num_epochs=conf.model.fusion_num_epochs)

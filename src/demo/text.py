from __future__ import print_function, division

from os import path
from warnings import filterwarnings

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
import datasets
import models
from models.cnn import TextSentiment

filterwarnings("ignore")

# Set the seed for pseudorandom operations
torch.manual_seed(conf.core.seed)
cudnn.deterministic = True

print('\nLoading data...')
text_dataset_saved_path = '/tmp/news_text_dataset.pth'
if path.exists(text_dataset_saved_path) and conf.core.load_text_dataset:
    text_dataset = torch.load(text_dataset_saved_path)
else:
    text_dataset = datasets.news.NewsTextDataset(conf.dataset.text_root_dir,
                                                 [conf.dataset.text_train_len,
                                                  conf.dataset.text_val_len,
                                                  conf.dataset.text_test_len])
    torch.save(text_dataset, text_dataset_saved_path)

image_dataloaders = {
    x: DataLoader(text_dataset.datasets[x],
                  batch_size=conf.dataset.text_batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'),
                  collate_fn=datasets.utils.generate_text_batch,
                  num_workers=0,
                  pin_memory=True)
    for x in conf.dataset.text_batch_sizes
}

VOCAB_SIZE = len(text_dataset.get_vocab())
EMBED_DIM = 32
NUM_CLASS = len(text_dataset.get_labels())

text_model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(conf.core.device)
text_criterion = torch.nn.CrossEntropyLoss().to(conf.core.device)
text_optimizer = torch.optim.SGD(text_model.parameters(), lr=4.0)
text_scheduler = torch.optim.lr_scheduler.StepLR(text_optimizer, 1, gamma=.9)

pipeline = models.TextTrainingPipeline(text_model, text_optimizer, text_criterion, text_scheduler)

pipeline.run(data_train, data_eval, data_test, conf.model.text_num_epochs)

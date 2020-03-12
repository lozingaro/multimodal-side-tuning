import os
from collections import OrderedDict

from torchtext.datasets import text_classification

import conf

from torch.utils.data import Dataset, DataLoader, random_split
import torch


class NewsTextDataset(Dataset):
    def __init__(self, root, lenghts):
        self.root = root
        self.lenghts = lenghts
        if not os.path.isdir(self.root):
            os.mkdir(self.root)

        self.original_train_dataset, self.test = text_classification.DATASETS['AG_NEWS'](
            root=self.root,
            ngrams=conf.dataset.text_ngrams,
            vocab=None)

        random_dataset_split = random_split(train_val, [self.lenghts[0], self.lenghts[1]])
        self.train = random_dataset_split[0]
        self.val = random_dataset_split[1]
        self.datasets = OrderedDict({
            'train': self.train,
            'val': self.val,
            'test': self.test
        })

    def get_vocab(self):
        return self.original_train_dataset.get_vocab()

    def get_labels(self):
        return self.original_train_dataset.get_labels()

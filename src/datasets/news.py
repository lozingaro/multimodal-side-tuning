import os
from collections import OrderedDict

from torch.utils.data import Dataset, random_split
from torchtext.datasets import text_classification

import conf


class NewsTextDataset(Dataset):
    def __init__(self, root, splits):
        self.root = root
        if not os.path.isdir(self.root):
            os.mkdir(self.root)

        self.original_train_dataset, self.test = text_classification.DATASETS['AG_NEWS'](
            root=self.root,
            ngrams=conf.dataset.text_ngrams,
            vocab=None)

        random_dataset_split = random_split(self.original_train_dataset, [splits['train'], splits['val']])
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

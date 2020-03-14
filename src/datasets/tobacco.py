import copy
import os
from collections import OrderedDict

import numpy as np
import spacy
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

import conf


class TobaccoImageDataset(Dataset):
    def __init__(self, root, splits=None):
        self.root = root
        full = datasets.ImageFolder(self.root)  # build a proper vision dataset
        self.imgs = copy.deepcopy(full.samples)
        self.extensions = full.extensions  # maybe useless?
        self.class_to_idx = full.class_to_idx  # maybe useless?
        self.samples = full.samples
        self.classes = full.classes
        self.targets = full.targets
        self.loader = full.loader

        if splits is None:
            self.lentghs = [800, 200, 2482]
        else:
            self.lentghs = splits.values()

        random_dataset_split = random_split(self, lengths=self.lentghs)
        self.train = random_dataset_split[0]
        self.val = random_dataset_split[1]
        self.test = random_dataset_split[2]
        self.datasets = OrderedDict({
            'train': self.train,
            'val': self.val,
            'test': self.test,
        })
        self.preprocess()

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def preprocess(self):
        w = conf.dataset.image_width
        i = conf.dataset.image_interpolation
        m = conf.dataset.image_mean_normalization
        s = conf.dataset.image_std_normalization
        t = {
            'train': transforms.Compose([
                transforms.Resize((w, w), interpolation=i),
                transforms.ToTensor(),
                transforms.Normalize(m, s),
            ]),
            'val': transforms.Compose([
                transforms.Resize((w, w), interpolation=i),
                transforms.ToTensor(),
                transforms.Normalize(m, s),
            ]),
            'test': transforms.Compose([
                transforms.Resize((w, w), interpolation=i),
                transforms.ToTensor(),
                transforms.Normalize(m, s),
            ]),
        }

        for index in range(len(self.samples)):
            path, target = self.samples[index]
            sample = self.loader(path)
            if index in self.train.indices:
                sample = t['train'](sample)
            elif index in self.val.indices:
                sample = t['val'](sample)
            elif index in self.test.indices:
                sample = t['test'](sample)
            self.samples[index] = (sample, target)

    def check_distributions(self):
        # check the distribution of full dataset
        partial_sums = np.unique(self.targets, return_counts=True)[1]
        partial_probs = [x / self.__len__() for x in partial_sums]
        # check the distribution of the train dataset
        partial_sums_train = np.unique([self.targets[i] for i in self.train.indices], return_counts=True)[1]
        partial_probs_train = [x / len(self.train) for x in partial_sums_train]
        # check the distribution of the val dataset
        partial_sums_val = np.unique([self.targets[i] for i in self.val.indices], return_counts=True)[1]
        partial_probs_val = [x / len(self.val) for x in partial_sums_val]
        # check the distribution of the test dataset
        partial_sums_test = np.unique([self.targets[i] for i in self.test.indices], return_counts=True)[1]
        partial_probs_test = [x / len(self.test) for x in partial_sums_test]

        plt.bar(self.classes, partial_sums)
        plt.plot(partial_probs, color='red', marker='o', linestyle='-', label='full')
        plt.plot(partial_probs_train, color='blue', marker='o', linestyle='-', label='train')
        plt.plot(partial_probs_val, color='green', marker='o', linestyle='-', label='validation')
        plt.plot(partial_probs_test, color='black', marker='o', linestyle='-', label='test')
        plt.legend()
        plt.xticks(list(range(10)), self.classes, rotation=20)
        plt.yscale('log')
        plt.grid(True)
        plt.title('class frequency distribution')


class TobaccoTextDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, splits=None, encoding='utf-8'):
        self.root_dir = root_dir
        self.encoding = encoding
        self.classes = []
        self.class_to_idx = {}
        self.targets = []
        self.texts = []
        self._populate()
        self.tokens = []
        self._tokenization()

        if splits is None:
            self.lentghs = [1, 1, 1]
        else:
            self.lentghs = splits.values()

        random_dataset_split = random_split(self, lengths=self.lentghs)
        self.train = random_dataset_split[0]
        self.val = random_dataset_split[1]
        self.test = random_dataset_split[2]
        self.datasets = OrderedDict({
            'train': self.train,
            'val': self.val,
            'test': self.test
        })

    def __getitem__(self, item):
        return [i.vector for i in self.tokens[item]], self.texts[item]  # TODO each document is a 500 x 300 vectors

    def __len__(self):
        return len(self.texts)

    def _populate(self):
        for root, dirs, files in os.walk(self.root_dir, topdown=True):
            for i, label in enumerate(dirs):
                self.classes.append(label)
                self.class_to_idx[label] = i
                for root_label, _, filenames in os.walk(os.path.join(self.root_dir, label), topdown=True):
                    for name in filenames:
                        with open(os.path.join(root_label, name), encoding=self.encoding) as f:
                            text = f.read().replace('\n', '')
                            self.texts.append(text)
                            self.targets.append(self.class_to_idx[label])

    def _tokenization(self):
        nlp = spacy.load('/data01/stefanopio.zingaro/datasets/en_vectors_crawl_lg')
        for i, doc in enumerate(nlp.pipe(self.texts, disable=['parser', 'tagger', 'ner'])):
            self.tokens.append([doc])


if __name__ == '__main__':
    # d = TobaccoImageDataset(conf.dataset.image_root_dir, conf.dataset.image_lengths)
    # d.check_distributions()
    # plt.show()
    d = TobaccoTextDataset('/data01/stefanopio.zingaro/datasets/QS-OCR-small')

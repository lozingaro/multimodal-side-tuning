import copy
import io
import os
from collections import OrderedDict

import fasttext
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch.utils.data
from torchvision import datasets, transforms


class TobaccoImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_width, image_interpolation, image_mean_norm, image_std_norm, splits=None):
        self.root = root
        self.image_width = image_width
        self.image_interpolation = image_interpolation
        self.image_mean_norm = image_mean_norm
        self.image_std_norm = image_std_norm
        if splits is None:
            self.lentghs = [800, 200, 2482]
        else:
            self.lentghs = splits.values()

        full = datasets.ImageFolder(self.root)  # builds a proper vision dataset
        self.imgs = copy.deepcopy(full.samples)
        self.extensions = full.extensions  # maybe useless?
        self.class_to_idx = full.class_to_idx  # maybe useless?
        self.samples = full.samples
        self.classes = full.classes
        self.targets = full.targets
        self.loader = full.loader

        random_dataset_split = torch.utils.data.random_split(self, lengths=self.lentghs)
        self.datasets = OrderedDict({
            'train': random_dataset_split[0],
            'val': random_dataset_split[1],
            'test': random_dataset_split[2],
        })
        self._preprocess()

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

    def __len__(self):
        return len(self.samples)

    def _preprocess(self):
        t = transforms.Compose([
                transforms.Resize((self.image_width, self.image_width), interpolation=self.image_interpolation),
                transforms.ToTensor(),
                transforms.Normalize(self.image_mean_norm, self.image_std_norm),
            ])
        for index in range(len(self.samples)):
            path, target = self.samples[index]
            sample = self.loader(path)
            sample = t(sample)
            self.samples[index] = sample

    def check_distributions(self):
        # check the distribution of full dataset
        partial_sums = np.unique(self.targets, return_counts=True)[1]
        partial_probs = [x / self.__len__() for x in partial_sums]
        # check the distribution of the train dataset
        partial_sums_train = np.unique([self.targets[i] for i in self.datasets['train'].indices], return_counts=True)[1]
        partial_probs_train = [x / len(self.datasets['train']) for x in partial_sums_train]
        # check the distribution of the val dataset
        partial_sums_val = np.unique([self.targets[i] for i in self.datasets['val'].indices], return_counts=True)[1]
        partial_probs_val = [x / len(self.datasets['val']) for x in partial_sums_val]
        # check the distribution of the test dataset
        partial_sums_test = np.unique([self.targets[i] for i in self.datasets['test'].indices], return_counts=True)[1]
        partial_probs_test = [x / len(self.datasets['test']) for x in partial_sums_test]

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
    def __init__(self, root_dir, context, num_grams=1, splits=None, encoding='utf-8', fasttext_model_path=None):
        self.root_dir = root_dir
        self.context = context
        self.num_grams = num_grams
        if splits is None:
            self.lentghs = [800, 200, 2482]
        else:
            self.lentghs = splits.values()
        self.encoding = encoding
        self.fasttext_model_path = fasttext_model_path
        if self.fasttext_model_path is not None:
            self.nlp = fasttext.load_model(self.fasttext_model_path)
            self.vocab = self.nlp.get_words()
        self.classes = []
        self.class_to_idx = {}
        self.targets = []
        self.samples = []
        self.replace = lambda c: c if c.isalnum() else ''
        self.vocab = set()
        # self.ngrams = []
        self._preprocess()
        self.token_to_idx = {token: i for i, token in enumerate(self.vocab)}

        random_dataset_split = torch.utils.data.random_split(self, lengths=self.lentghs)
        self.train = random_dataset_split[0]
        self.val = random_dataset_split[1]
        self.test = random_dataset_split[2]
        self.datasets = OrderedDict({
            'train': self.train,
            'val': self.val,
            'test': self.test
        })

    def __getitem__(self, index):
        if self.fasttext_model_path is not None:
            tokens_tensor = torch.tensor([self.nlp.get_word_id(t) for t in self.samples[index]], dtype=torch.long)
        else:
            tokens_tensor = torch.tensor([self.token_to_idx[t] for t in self.samples[index]], dtype=torch.long)

        if len(self.samples[index]) < self.context:
            tokens_tensor = F.pad(tokens_tensor, (0, self.context - len(self.samples[index])))
        else:
            tokens_tensor = tokens_tensor[:self.context]

        return tokens_tensor, self.targets[index]

    def __len__(self):
        return len(self.targets)

    def _preprocess(self):
        for root, dirs, files in os.walk(self.root_dir, topdown=True):
            for i, label in enumerate(dirs):
                self.classes.append(label)
                self.class_to_idx[label] = i
                for root_label, _, filenames in os.walk(os.path.join(self.root_dir, label), topdown=True):
                    for name in filenames:
                        self._load_tokens(os.path.join(root_label, name))
                        self.targets.append(self.class_to_idx[label])

    def _load_tokens(self, fname):
        with io.open(fname, 'r', encoding=self.encoding, newline='\n', errors='ignore') as fin:
            tokens = fin.read().lower().split()

        tokens = [''.join([self.replace(c) for c in token]) for token in tokens]
        tokens = [token for token in tokens if len(token) > 0]
        if self.fasttext_model_path is None:
            self.vocab.update(set(tokens))
        # self.ngrams.append([[tokens[i + j] for j in range(self.num_grams)]
        #                     for i in range(len(tokens) - self.num_grams - 1)])
        self.samples.append(tokens)


if __name__ == '__main__':
    # d = TobaccoImageDataset(conf.dataset.image_root_dir, conf.dataset.image_lengths)
    # d.check_distributions()
    # plt.show()
    d = TobaccoTextDataset('/data01/stefanopio.zingaro/datasets/QS-OCR-small')
    e = d[0]
    print(e[0].size())

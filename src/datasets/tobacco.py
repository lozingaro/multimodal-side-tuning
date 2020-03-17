import copy
import io
import os
from collections import OrderedDict, Counter

import numpy as np
import spacy
import torch
import torch.nn.functional as F
import torch.utils.data
from matplotlib import pyplot as plt
from torchvision import datasets, transforms


class TobaccoFusionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dataset, text_dataset, splits=None):
        super(TobaccoFusionDataset, self).__init__()
        if splits is None:
            self.lengths = [800, 200, 2482]
        else:
            self.lengths = splits.values()
        self.image_dataset = image_dataset
        self.text_dataset = text_dataset
        random_dataset_split = torch.utils.data.random_split(self, lengths=self.lengths)
        self.datasets = OrderedDict({
            'train': random_dataset_split[0],
            'val': random_dataset_split[1],
            'test': random_dataset_split[2]
        })

    def __getitem__(self, index):
        out_image = self.image_dataset[index]
        out_text = self.text_dataset[index][0]
        return (out_image[0], out_text), out_image[1]

    def __len__(self):
        return len(self.image_dataset.targets)


class TobaccoImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_width, image_interpolation, image_mean_norm, image_std_norm, splits=None):
        self.root = root
        self.image_width = image_width
        self.image_interpolation = image_interpolation
        self.image_mean_norm = image_mean_norm
        self.image_std_norm = image_std_norm
        if splits is None:
            self.lengths = [800, 200, 2482]
        else:
            self.lengths = splits.values()

        full = datasets.ImageFolder(self.root)  # builds a proper vision dataset
        self.imgs = copy.deepcopy(full.samples)
        self.extensions = full.extensions  # maybe useless?
        self.class_to_idx = full.class_to_idx  # maybe useless?
        self.samples = full.samples
        self.classes = full.classes
        self.targets = full.targets
        self.loader = full.loader

        random_dataset_split = torch.utils.data.random_split(self, lengths=self.lengths)
        self.datasets = OrderedDict({
            'train': random_dataset_split[0],
            'val': random_dataset_split[1],
            'test': random_dataset_split[2],
        })
        self._preprocess()

    def __getitem__(self, index):
        return self.samples[index], torch.tensor(self.targets[index], dtype=torch.long)

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
    def __init__(self, root_dir, context, num_grams=1, splits=None, encoding='utf-8', nlp_model_path=None):
        self.root_dir = root_dir
        self.context = context
        self.num_grams = num_grams
        if splits is None:
            self.lengths = [800, 200, 2482]
        else:
            self.lengths = splits.values()
        self.encoding = encoding
        self.nlp_model_path = nlp_model_path
        self.classes = []
        self.class_to_idx = {}
        self.targets = []
        self.texts = []
        self.tokens = []
        if self.nlp_model_path is None:
            self.nlp = None
            self.vocab = set()
        else:
            self.nlp = spacy.load(self.nlp_model_path)
            self.vocab = self.nlp.vocab
        self._preprocess()
        if self.nlp_model_path is None:
            self.token_to_idx = {token: i for i, token in enumerate(self.vocab)}
        self.samples = []
        self._load_tensors()
        random_dataset_split = torch.utils.data.random_split(self, lengths=self.lengths)
        self.datasets = OrderedDict({
            'train': random_dataset_split[0],
            'val': random_dataset_split[1],
            'test': random_dataset_split[2]
        })

    def __getitem__(self, index):
        return self.samples[index], torch.tensor(self.targets[index], dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def _load_tensors(self):
        for tokens in self.tokens:
            padding = self.context - len(tokens)
            if self.nlp_model_path is not None:
                if len(tokens) == 0:
                    tokens_tensor = torch.zeros((500, 300))
                else:
                    tokens_tensor = torch.tensor([t.vector for t in tokens])
                    if len(tokens) < self.context:
                        tokens_tensor = F.pad(tokens_tensor, (0, 0, 0, padding))
                    else:
                        tokens_tensor = tokens_tensor[:self.context]
            else:
                if len(tokens) == 0:
                    tokens_tensor = torch.zeros(500, dtype=torch.long)
                else:
                    tokens_tensor = torch.tensor([self.token_to_idx[t] for t in tokens], dtype=torch.long)
                    tokens_tensor = F.pad(tokens_tensor, (0, padding))

            self.samples.append(tokens_tensor)

    def _preprocess(self):
        replace = lambda c: c
        for root, dirs, files in os.walk(self.root_dir, topdown=True):
            for i, label in enumerate(dirs):
                self.classes.append(label)
                self.class_to_idx[label] = i
                for root_label, _, filenames in os.walk(os.path.join(self.root_dir, label), topdown=True):
                    for name in filenames:
                        self.texts.append(os.path.join(self.root_dir, label, name))
                        self._load_tokens(os.path.join(root_label, name), replace)
                        self.targets.append(self.class_to_idx[label])

    def _load_tokens(self, fname, replace):
        with io.open(fname, 'r', encoding=self.encoding, newline='\n', errors='ignore') as fin:
            doc = fin.read().replace('\n', ' ').lower()

        if self.nlp_model_path is not None:
            doc = self.nlp(doc)
            tokens = [token for token, (word, freq) in
                      zip([token for token in doc], Counter([token.text for token in doc]).items())
                      if not token.is_punct and not token.is_stop and freq == 1 and len(token) > 0]
        else:
            doc = doc.split()
            tokens = [''.join([replace(c) for c in token]) for token in doc if len(token) > 0]
            self.vocab.update(set(tokens))

        self.tokens.append(tokens)


if __name__ == '__main__':
    # d = TobaccoImageDataset(conf.dataset.image_root_dir, conf.dataset.image_lengths)
    # d.check_distributions()
    # plt.show()
    d = TobaccoTextDataset('/data01/stefanopio.zingaro/datasets/QS-OCR-small')
    e = d[0]
    print(e[0].size())

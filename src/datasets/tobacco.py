import copy
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split
from torchtext.datasets import text_classification
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


class TobaccoTextDataset(Dataset):
    def __init__(self, root, splits):
        self.root = root
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


if __name__ == '__main__':
    d = TobaccoImageDataset(conf.dataset.image_root_dir, conf.dataset.image_lengths)
    d.check_distributions()
    plt.show()

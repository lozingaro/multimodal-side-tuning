import copy
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

from conf import dataset


class TobaccoImageDataset(Dataset):
    def __init__(self, root, lengths=None):
        self.root = root
        full = datasets.ImageFolder(self.root)  # build a proper vision dataset
        self.imgs = copy.deepcopy(full.samples)
        self.extensions = full.extensions  # maybe useless?
        self.class_to_idx = full.class_to_idx  # maybe useless?
        self.samples = full.samples
        self.classes = full.classes
        self.targets = full.targets
        self.loader = full.loader

        if lengths is None:
            self.lentghs = [800, 200, 2482]
        else:
            self.lentghs = lengths

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
        w = dataset.image_width
        i = dataset.image_interpolation
        m = dataset.image_mean_normalization
        s = dataset.image_std_normalization
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


if __name__ == '__main__':
    d = TobaccoImageDataset(dataset.image_root_dir, list(dataset.image_len.values()))
    d.check_distributions()
    plt.show()

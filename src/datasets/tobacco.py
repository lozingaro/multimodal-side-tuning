import numpy as np

from torch.utils.data import Dataset, random_split
from torchvision import datasets


class TobaccoImageDataset(Dataset):
    def __init__(self, root, lengths=None, transforms=None):
        self.root = root
        full = datasets.ImageFolder(self.root)  # build a proper vision dataset

        self.imgs = full.imgs  # maybe useless?
        self.extensions = full.extensions  # maybe useless?
        self.class_to_idx = full.class_to_idx  # maybe useless?
        self.samples = full.samples
        self.classes = full.classes
        self.targets = full.targets
        self.loader = full.loader

        if lengths is None:
            self.lentghs = [self.__len__(), 0, 0]
        else:
            self.lentghs = lengths
        self.transforms = transforms

        random_dataset_split = random_split(full, lengths=self.lentghs)
        self.train = random_dataset_split[0]
        self.val = random_dataset_split[1]
        self.test = random_dataset_split[2]

        if self.transforms is not None:
            for index in range(len(self.samples)):
                path, target = self.samples[index]
                sample = self.loader(path)
                transforms = []
                if index in self.train.indices:
                    transforms = self.transforms['train']
                elif index in self.val.indices:
                    transforms = self.transforms['val']
                elif index in self.test.indices:
                    transforms = self.transforms['test']
                sample = transforms(sample)
                self.samples[index] = (sample, target)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def check_distributions(self):
        # check the distribution of full dataset
        partial_sums = np.unique(self.targets, return_counts=True)[1]
        partial_probs = [x / self.__len__() for x in partial_sums]
        # check the distribution of the train dataset
        partial_sums_train = np.unique([self.targets[i] for i in self.train.indices], return_counts=True)[1]
        partial_probs_train = [x / len(self.train) for x in partial_sums_train]
        print(partial_probs, partial_probs_train)

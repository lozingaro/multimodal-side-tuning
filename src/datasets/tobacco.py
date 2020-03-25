import io
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms


class TobaccoDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(TobaccoDataset, self).__init__()
        if self.targets is None:
            self.targets = []
        self.lengths = [800, 200, 2482]
        self.datasets = self._load_datasets()

    def _load_datasets(self):
        random_dataset_split = torch.utils.data.random_split(self, self.lengths)
        return OrderedDict({
            'train': random_dataset_split[0],
            'val': random_dataset_split[1],
            'test': random_dataset_split[2]
        })

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


class FusionDataset(TobaccoDataset):
    def __init__(self, image_dataset, text_dataset):
        self.image_dataset = image_dataset
        self.text_dataset = text_dataset
        self.classes = self.image_dataset.classes
        self.samples = self._load_samples()
        self.targets = self._load_targets()
        super(FusionDataset, self).__init__()

    def _load_samples(self):
        samples = []
        for image, text in zip(self.image_dataset, self.text_dataset):
            samples.append((image[0], text[0]))
        return samples

    def _load_targets(self):
        return self.image_dataset.targets


class ImageDataset(TobaccoDataset):
    def __init__(self, root, image_width, image_interpolation, image_mean_norm, image_std_norm):
        self.full = datasets.ImageFolder(root)
        self.image_width = image_width
        self.image_interpolation = image_interpolation
        self.image_mean_norm = image_mean_norm
        self.image_std_norm = image_std_norm
        self.classes = self.full.classes
        self.samples = self._load_samples()
        self.targets = self.full.targets
        super(ImageDataset, self).__init__()

    def _load_samples(self):
        samples = []
        t = transforms.Compose([
            transforms.Resize((self.image_width, self.image_width), interpolation=self.image_interpolation),
            transforms.ToTensor(),
            transforms.Normalize(self.image_mean_norm, self.image_std_norm),
        ])
        for index in range(len(self.full.samples)):
            path, target = self.full.samples[index]
            sample = self.full.loader(path)
            samples.append(t(sample))
        return samples


class TextDataset(TobaccoDataset):
    def __init__(self, root, nlp, context=500):
        self.root = root
        self.nlp = nlp
        self.context = context
        self.targets = self._load_targets()
        self.samples = self._load_samples()
        super(TextDataset, self).__init__()

    def _load_targets(self):
        self.classes = []
        targets = []
        self.texts = []
        for root, dirs, _ in os.walk(self.root):
            for i, label in enumerate(dirs):
                self.classes.append(label)
                for root_label, _, filenames in os.walk(os.path.join(root, label)):
                    self.texts += [os.path.join(root_label, name) for name in filenames]
                    targets += [i for _ in filenames]
        return targets

    def _load_samples(self):
        samples = []
        for fname in self.texts:
            with io.open(fname, encoding='utf-8') as f:
                doc = f.read()
            doc = [self.nlp[i] for i in doc.split()]
            padding = self.context - len(doc)
            if padding > 0:
                if padding == 500:
                    samples.append(torch.zeros((500, 300)))
                else:
                    samples.append(F.pad(torch.tensor(doc), [0, 0, 0, padding]))
            else:
                samples.append(torch.tensor(doc[:self.context]))
        return samples

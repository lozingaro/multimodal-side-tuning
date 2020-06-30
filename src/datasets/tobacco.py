from __future__ import division, print_function

import os
import random
from warnings import filterwarnings

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader

filterwarnings("ignore")
cudnn.deterministic = True
cudnn.benchmark = False

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class TobaccoDataset(torch.utils.data.Dataset):
    def __init__(self, img_root_dir, txt_root_dir):
        super(TobaccoDataset, self).__init__()
        self.targets = []
        self.imgs = []
        self.txts = []
        for i, (txt_class_path, img_class_path) in enumerate(zip(os.scandir(txt_root_dir), os.scandir(img_root_dir))):
            for txt_path, img_path in zip(os.scandir(txt_class_path), os.scandir(img_class_path)):
                self.targets += [i]
                self.imgs += [img_path.path]
                self.txts += [txt_path.path]

    def __getitem__(self, item):
        img = tf.to_tensor(Image.open(self.imgs[item]))
        txt = torch.load(self.txts[item]).float()
        return (img, txt), self.targets[item]

    def __len__(self):
        return len(self.targets)


class TobaccoImgDataset(torch.utils.data.Dataset):
    def __init__(self, img_root_dir):
        super(TobaccoImgDataset, self).__init__()
        self.targets = []
        self.imgs = []
        for i, img_class_path in enumerate(os.scandir(img_root_dir)):
            for img_path in os.scandir(img_class_path):
                self.targets += [i]
                self.imgs += [img_path.path]

    def __getitem__(self, item):
        img = tf.to_tensor(Image.open(self.imgs[item]))
        return img, self.targets[item]

    def __len__(self):
        return len(self.targets)


class TobaccoTxtDataset(torch.utils.data.Dataset):
    def __init__(self, txt_root_dir):
        super(TobaccoTxtDataset, self).__init__()
        self.targets = []
        self.txts = []
        for i, txt_class_path in enumerate(os.scandir(txt_root_dir)):
            for txt_path in os.scandir(txt_class_path):
                self.targets += [i]
                self.txts += [txt_path.path]

    def __getitem__(self, item):
        txt = torch.load(self.txts[item]).float()
        return txt, self.targets[item]

    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':
    img_dataset_dir = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/Tobacco3482-jpg'
    txt_dataset_dir = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/QS-OCR-small'
    d = TobaccoDataset(f'{img_dataset_dir}', f'{txt_dataset_dir}')
    r = torch.utils.data.random_split(d, [800, 200, 2482])
    d_train = r[0]
    d_val = r[1]
    d_test = r[2]
    dl_train = DataLoader(d_train, batch_size=16, shuffle=True)
    dl_val = DataLoader(d_val, batch_size=4, shuffle=True)
    dl_test = DataLoader(d_test, batch_size=32, shuffle=False)

"""
    Multimodal side-tuning for document classification
    Copyright (C) 2020  S.P. Zingaro <mailto:stefanopio.zingaro@unibo.it>.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import division, print_function

import os
import random
from warnings import filterwarnings

import numpy as np
import torch
import torchvision.transforms.functional as F
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
        self.classes = []
        self.targets = []
        self.imgs = []
        self.txts = []
        for i, (txt_class_path, img_class_path) in enumerate(zip(os.scandir(txt_root_dir), os.scandir(img_root_dir))):
            self.classes += [txt_class_path.name]
            for txt_path, img_path in zip(os.scandir(txt_class_path), os.scandir(img_class_path)):
                self.targets += [i]
                self.imgs += [img_path.path]
                self.txts += [txt_path.path]

    def __getitem__(self, item):
        img = F.to_tensor(Image.open(self.imgs[item]))
        txt = torch.load(self.txts[item]).float()
        return (img, txt), self.targets[item]

    def __len__(self):
        return len(self.targets)


class TobaccoImgDataset(torch.utils.data.Dataset):
    def __init__(self, img_root_dir):
        super(TobaccoImgDataset, self).__init__()
        self.classes = []
        self.targets = []
        self.imgs = []
        for i, img_class_path in enumerate(os.scandir(img_root_dir)):
            self.classes += [img_class_path.name]
            for img_path in os.scandir(img_class_path):
                self.targets += [i]
                self.imgs += [img_path.path]

    def __getitem__(self, item):
        img = F.to_tensor(Image.open(self.imgs[item]))
        return img, self.targets[item]

    def __len__(self):
        return len(self.targets)


class TobaccoTxtDataset(torch.utils.data.Dataset):
    def __init__(self, txt_root_dir):
        super(TobaccoTxtDataset, self).__init__()
        self.classes = []
        self.targets = []
        self.txts = []
        for i, txt_class_path in enumerate(os.scandir(txt_root_dir)):
            self.classes += [txt_class_path.name]
            for txt_path in os.scandir(txt_class_path):
                self.targets += [i]
                self.txts += [txt_path.path]

    def __getitem__(self, item):
        txt = torch.load(self.txts[item]).float()
        return txt, self.targets[item]

    def __len__(self):
        return len(self.targets)

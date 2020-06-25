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

from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F


class TobaccoDataset(torch.utils.data.Dataset):
    def __init__(self, img_root_dir, txt_root_dir):
        super(TobaccoDataset, self).__init__()
        self.classes = []
        self.targets = []
        self.imgs = []
        self.txts = []
        for i, label in enumerate(sorted(os.listdir(txt_root_dir))):
            txt_class_path = f'{txt_root_dir}/{label}'
            img_class_path = f'{img_root_dir}/{label}'
            self.classes += [label]
            for txt_path in os.scandir(txt_class_path):
                self.targets += [i]
                img = Image.open(f'{img_class_path}/{".".join(txt_path.name.split(".")[:-1])}.jpg')
                img = F.to_tensor(img)
                img = F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                self.imgs += [img]
                txt = torch.load(txt_path.path).float()
                self.txts += [txt]

    def __getitem__(self, item):
        return (self.imgs[item], self.txts[item]), self.targets[item]

    def __len__(self):
        return len(self.targets)


class TobaccoImgDataset(torch.utils.data.Dataset):
    def __init__(self, img_root_dir):
        super(TobaccoImgDataset, self).__init__()
        self.classes = []
        self.targets = []
        self.imgs = []
        for i, label in enumerate(sorted(os.listdir(img_root_dir))):
            img_class_path = f'{img_root_dir}/{label}'
            self.classes += [label]
            with os.scandir(img_class_path) as it:
                for img_path in it:
                    self.targets += [i]
                    img = Image.open(img_path.path)
                    img = F.to_tensor(img)
                    img = F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    self.imgs += [img]

    def __getitem__(self, item):
        return self.imgs[item], self.targets[item]

    def __len__(self):
        return len(self.targets)


class TobaccoTxtDataset(torch.utils.data.Dataset):
    def __init__(self, txt_root_dir):
        super(TobaccoTxtDataset, self).__init__()
        self.classes = []
        self.targets = []
        self.txts = []
        for i, label in enumerate(sorted(os.listdir(txt_root_dir))):
            txt_class_path = f'{txt_root_dir}/{label}'
            self.classes += [label]
            with os.scandir(txt_class_path) as it:
                for txt_path in it:
                    self.targets += [i]
                    txt = torch.load(txt_path.path).float()
                    self.txts += [txt]

    def __getitem__(self, item):
        return self.txts[item], self.targets[item]

    def __len__(self):
        return len(self.targets)

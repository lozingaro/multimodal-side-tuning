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
import torchvision.transforms.functional as TF
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
        label_to_target = {}
        i = 0
        for label in sorted(os.listdir(img_root_dir)):
            if os.path.isdir(f'{img_root_dir}/{label}'):
                txt_class_path = f'{txt_root_dir}/{label}'
                img_class_path = f'{img_root_dir}/{label}'
                self.classes += [label]
                label_to_target[label] = i
                i += 1
                for img_path in os.scandir(img_class_path):
                    filename, file_extension = os.path.splitext(img_path.name)
                    if file_extension.lower() in ['.tif', '.jpg']:
                        txt_path = f'{txt_class_path}/{filename}.txt'
                        if not os.path.isfile(txt_path) or not os.path.isfile(img_path.path):
                            raise FileNotFoundError(f'Did not find {txt_path} or {img_path.path}!')
                        self.targets += [label_to_target[label]]
                        self.imgs += [img_path.path]
                        self.txts += [txt_path]

    def __getitem__(self, item):
        img = TF.to_tensor(Image.open(self.imgs[item]))
        # txt = torch.load(self.txts[item]).float()
        return (img, self.txts[item]), self.targets[item]

    def __len__(self):
        return len(self.targets)


def split_tobacco(split_dir):
    def move(source, dest):
        from subprocess import call
        call(["cp", source, dest])
        return None
    d = TobaccoDataset(img_root_dir='/Volumes/SD128/Developer/ocrized-text-dataset/datasets/Tobacco3482', txt_root_dir='/Volumes/SD128/Developer/ocrized-text-dataset/datasets/QS-OCR-small')
    d_train, d_val, d_test = torch.utils.data.random_split(d, [800, 200, 2482])
    if not os.path.isdir(split_dir):
        os.mkdir(split_dir)
        os.mkdir(f'{split_dir}/train')
        os.mkdir(f'{split_dir}/train/img')
        os.mkdir(f'{split_dir}/train/txt')
        os.mkdir(f'{split_dir}/val')
        os.mkdir(f'{split_dir}/val/img')
        os.mkdir(f'{split_dir}/val/txt')
        os.mkdir(f'{split_dir}/test')
        os.mkdir(f'{split_dir}/test/img')
        os.mkdir(f'{split_dir}/test/txt')
    for i in d_train.indices:
        l = d.classes[d[i][1]]
        if not os.path.isdir(f'{split_dir}/train/img/{l}'):
            os.mkdir(f'{split_dir}/train/img/{l}')
        if not os.path.isdir(f'{split_dir}/train/txt/{l}'):
            os.mkdir(f'{split_dir}/train/txt/{l}')
        move(d.imgs[i], f'{split_dir}/train/img/{l}/.')
        move(d.txts[i], f'{split_dir}/train/txt/{l}/.')
    for i in d_val.indices:
        l = d.classes[d[i][1]]
        if not os.path.isdir(f'{split_dir}/val/img/{l}'):
            os.mkdir(f'{split_dir}/val/img/{l}')
        if not os.path.isdir(f'{split_dir}/val/txt/{l}'):
            os.mkdir(f'{split_dir}/val/txt/{l}')
        move(d.imgs[i], f'{split_dir}/val/img/{l}/.')
        move(d.txts[i], f'{split_dir}/val/txt/{l}/.')
    for i in d_test.indices:
        l = d.classes[d[i][1]]
        if not os.path.isdir(f'{split_dir}/test/img/{l}'):
            os.mkdir(f'{split_dir}/test/img/{l}')
        if not os.path.isdir(f'{split_dir}/test/txt/{l}'):
            os.mkdir(f'{split_dir}/test/txt/{l}')
        move(d.imgs[i], f'{split_dir}/test/img/{l}/.')
        move(d.txts[i], f'{split_dir}/test/txt/{l}/.')


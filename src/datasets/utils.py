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

import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from torchtext.vocab import FastText, GloVe
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from torch.backends import cudnn
from warnings import filterwarnings

filterwarnings("ignore")
cudnn.deterministic = True
cudnn.benchmark = False

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def load_txt_samples(orig_dir, dest_dir, nlp):
    for label in sorted(os.listdir(orig_dir)):
        class_path = f'{orig_dir}/{label}'
        with os.scandir(class_path) as it:
            for _, path in tqdm(enumerate(it)):
                with open(path, 'rb') as f:
                    txt = f.read()
                doc = [''.join([i for i in token.decode('UTF-8') if i.isalnum()]) for token in txt.split()]
                word2vec = [nlp[i] for i in doc]
                padding = 500 - len(word2vec)
                if padding > 0:
                    if padding == 500:
                        x = torch.zeros((500, 300))
                    else:
                        x = F.pad(torch.tensor(word2vec), [0, 0, 0, padding])
                else:
                    x = torch.tensor(word2vec[:500])

                if not os.path.exists(f'{dest_dir}/{label}'):
                    os.mkdir(f'{dest_dir}/{label}')
                torch.save(x.half(), f'{dest_dir}/{label}/{"".join(path.name.split(".")[:-1])}.ptr')


def load_img_samples(orig_dir, dest_dir):
    for label in sorted(os.listdir(orig_dir)):
        class_path = f'{orig_dir}/{label}'
        with os.scandir(class_path) as it:
            for _, path in tqdm(enumerate(it)):
                with open(path, 'rb') as f:
                    try:
                        img = Image.open(f)
                        img = img.convert('RGB')
                        img = img.resize((384, 384))
                        if not os.path.exists(f'{dest_dir}/{label}'):
                            os.mkdir(f'{dest_dir}/{label}')
                        img.save(f'{dest_dir}/{label}/{"".join(path.name.split(".")[:-1])}.jpg', "JPEG", quality=100)
                    except UnidentifiedImageError:
                        pass


if __name__ == '__main__':
    fasttext_model = FastText()
    glove_model = GloVe()
    load_img_samples('../data/original/Tobacco3482-jpg',
                     '../data/Tobacco3482-jpg')
    load_txt_samples('../data/original/QS-OCR-small',
                     '../data/QS-OCR-small', fasttext_model)
    for s in ['val', 'test', 'train']:
        load_img_samples(f'../data/original/RVL-CDIP/{s}',
                         f'../data/RVL-CDIP/{s}')
        load_txt_samples(f'../data/original/QS-OCR-Large/{s}',
                         f'../data/QS-OCR-Large/{s}', fasttext_model)

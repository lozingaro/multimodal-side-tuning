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

import itertools
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tobacco_img_root_dir = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/Tobacco3482-jpg'
tobacco_txt_root_dir = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/QS-OCR-small'
rlv_img_root_dir = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/RVL-CDIP'
rlv_txt_root_dir = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/QS-OCR-Large'
text_fasttext_model_path = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/cc.en.300.bin'

tasks_classifier = ['direct', '1280x128x10', '1280x256x10', '1280x512x10', '1280x1024x10', 'concat', ]
tasks_optimizer = ['sgd', 'adam']
tasks_embedding = ['fasttext', 'custom']
tasks_loss_weigth = ['min', 'max', 'no']
tasks_coeffs = ['4-3-3', '5-3-2', '4-4-2', '3-3-4', '2-4-4', '2-3-5']
tasks = itertools.product(tasks_classifier, tasks_optimizer, tasks_embedding, tasks_loss_weigth, tasks_coeffs)

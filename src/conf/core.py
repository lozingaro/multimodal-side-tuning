"""
LICENSE and explanation
"""

import itertools
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tobacco_img_root_dir = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/Tobacco3482-jpg'
tobacco_txt_root_dir = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/QS-OCR-small'
rlv_img_root_dir = '/data01/stefanopio.zingaro/datasets/RVL-CDIP'
rlv_txt_root_dir = '/data01/stefanopio.zingaro/datasets/QS-OCR-Large'
text_fasttext_model_path = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/cc.en.300.bin'

tasks_classifier = ['1280x512x10', '1280x1024x10']  # ['direct', '1280x128x10', '1280x256x10', '1280x512x10', '1280x1024x10', 'concat', ]
tasks_optimizer = ['sgd']  # ['sgd', 'adam']
tasks_embedding = ['fasttext']  # ['fasttext', 'custom']
tasks_loss_weigth = ['min']
tasks_coeffs = ['2-4-4', '2-3-5']  # ['4-3-3', '5-3-2', '4-4-2', '3-3-4']
tasks = itertools.product(tasks_classifier, tasks_optimizer, tasks_embedding, tasks_loss_weigth, tasks_coeffs)

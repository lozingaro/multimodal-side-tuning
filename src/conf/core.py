import itertools

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_root_dir = '/data01/stefanopio.zingaro/datasets/original/Tobacco3482-jpg'
text_root_dir = '/data01/stefanopio.zingaro/datasets/original/QS-OCR-small'
batch_sizes = {
    'train': 16,
    'val': 4,
    'test': 32
}
text_fasttext_model_path = '/data01/stefanopio.zingaro/datasets/fasttext/cc.en.300.bin'
text_dataset_custom_path = '/tmp/text_dataset_custom.pth'
text_dataset_fasttext_path = '/tmp/text_dataset_fasttext.pth'
image_dataset_path = '/tmp/image_dataset.pth'
fusion_dataset_custom_path = '/tmp/fusion_dataset_custom.pth'
fusion_dataset_fasttext_path = '/tmp/fusion_dataset_fasttext.pth'

tasks_classifier = ['1280x1024x10']  # ['1280x10', '1280x512x10', '1280x128x10']
tasks_optimizer = ['sgd']  # ['sgd', 'adam']
tasks_embedding = ['fasttext']  # ['fasttext', 'custom']
tasks_loss_weigth = ['no']  # ['no', 'min', 'max']
tasks_coeffs = ['3-3-4']  # ['4-3-3', '5-3-2', '4-4-2']
tasks = itertools.product(tasks_classifier, tasks_optimizer, tasks_embedding, tasks_loss_weigth, tasks_coeffs)

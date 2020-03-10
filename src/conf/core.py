import os
import torch
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_dir = '/data01/stefanopio.zingaro/datasets/Tobacco3482-jpg'
seed = 42
batch_sizes = {
    'train': 16,
    'val': 4,
    'test': 128,
}
lengths = {
    'train': 800,
    'val': 200,
    'test': 2482,
}
load_dataset = False
workers = os.cpu_count() if not None else 0
image_mean_normalization = [0.485, 0.456, 0.406]
image_std_normalization = [0.229, 0.224, 0.225]
image_interpolation = Image.BILINEAR
image_width = 224

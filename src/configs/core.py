import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_dir = '/Users/zingaro/SD128/Developer/multimodal-side-tuning/data/Tobacco3482-jpg'
# root_dir = '/data01/stefanopio.zingaro/my_datasets/tobacco-image'
batch_sizes = {
    'train': 4,
    'val': 1,
    'test': 1,
}
lengths = {
    'train': 800,
    'val': 200,
    'test': 3482 - 800 - 200,
}
build_dataset_from_scratch = False
workers = os.cpu_count() if not None else 0
mean_normalization = [0.485, 0.456, 0.406]
std_normalization = [0.229, 0.224, 0.225]

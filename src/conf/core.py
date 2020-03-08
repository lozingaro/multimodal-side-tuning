import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# root_dir = '/Users/zingaro/SD128/Developer/multimodal-side-tuning/data/Tobacco3482-jpg'
root_dir = '/data01/stefanopio.zingaro/datasets/Tobacco3482-jpg'
batch_sizes = {
    'train': 16,
    'val': 4,
    'test': 2482,
}
lengths = {
    'train': 800,
    'val': 200,
    'test': 2482,
}
build_dataset_from_scratch = True
build_model_from_scratch = True
workers = os.cpu_count() if not None else 0
mean_normalization = [0.485, 0.456, 0.406]
std_normalization = [0.229, 0.224, 0.225]

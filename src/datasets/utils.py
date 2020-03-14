import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import make_grid

import conf


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(conf.dataset.image_mean_normalization)
    std = np.array(conf.dataset.image_std_normalization)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def generate_text_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


if __name__ == '__main__':
    # Visualize a few images
    t = transforms.Compose([
        transforms.Resize((conf.dataset.image_width, conf.dataset.image_width)),
        transforms.ToTensor(),
        transforms.Normalize(conf.dataset.image_mean_normalization, conf.dataset.image_std_normalization),
    ])
    d = ImageFolder(root='/data01/stefanopio.zingaro/datasets/Tobacco3482-jpg', transform=t)
    dataloader = DataLoader(d, batch_size=conf.dataset.batch_sizes['train'], shuffle=True)
    one_batch_inputs, one_batch_classes = next(iter(dataloader))
    out = make_grid(one_batch_inputs, nrow=conf.dataset.batch_sizes['train'] // 4)
    imshow(out)

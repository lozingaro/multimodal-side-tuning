import random

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from conf import core


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(core.image_mean_normalization)
    std = np.array(core.image_std_normalization)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def lr_pad(img):
    return int(abs(img.height - core.image_width) // 2)


def l_pad(img):
    if abs(img.height - core.image_width) % 2 == 0:
        return lr_pad(img)
    else:
        return lr_pad(img) + 1


def r_pad(img):
    return lr_pad(img)


def degree():
    r = random.uniform(0, 100)
    p = [50, 66, 82]
    d = [0, 90, 180, 270]
    if r < p[0]:
        return d[0]
    elif r < p[1]:
        return d[1]
    elif r < p[2]:
        return d[2]
    else:
        return d[3]


if __name__ == '__main__':
    # Visualize a few images
    t = transforms.Compose([
        transforms.Resize((core.image_width, core.image_width)),
        transforms.ToTensor(),
        transforms.Normalize(core.image_mean_normalization, core.image_std_normalization),
    ])
    d = ImageFolder(root='/data01/stefanopio.zingaro/datasets/Tobacco3482-jpg', transform=t)
    dataloader = DataLoader(d, batch_size=core.batch_sizes['train'], shuffle=True)
    one_batch_inputs, one_batch_classes = next(iter(dataloader))
    out = make_grid(one_batch_inputs, nrow=core.batch_sizes['train'] // 4)
    imshow(out)

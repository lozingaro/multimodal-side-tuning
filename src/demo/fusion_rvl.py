from __future__ import division, print_function

import random
from collections import OrderedDict
from warnings import filterwarnings

import fasttext as fasttext
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.tobacco import FusionDataset, ImageDataset, TextDataset
from models import TrainingPipeline, FusionSideNetSideFC

filterwarnings("ignore")
cudnn.deterministic = True
cudnn.benchmark = False

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

try:
    image_dataset = torch.load("/tmp/image_dataset_rvl.pth")
except FileNotFoundError:
    train_image_dataset = ImageDataset('/data01/stefanopio.zingaro/datasets/RVL-CDIP/train',
                                       image_width=384,
                                       image_interpolation=Image.BILINEAR,
                                       image_mean_norm=[0.485, 0.456, 0.406],
                                       image_std_norm=[0.229, 0.224, 0.225])
    val_image_dataset = ImageDataset('/data01/stefanopio.zingaro/datasets/RVL-CDIP/val',
                                     image_width=384,
                                     image_interpolation=Image.BILINEAR,
                                     image_mean_norm=[0.485, 0.456, 0.406],
                                     image_std_norm=[0.229, 0.224, 0.225])
    test_image_dataset = ImageDataset('/data01/stefanopio.zingaro/datasets/RVL-CDIP/test',
                                      image_width=384,
                                      image_interpolation=Image.BILINEAR,
                                      image_mean_norm=[0.485, 0.456, 0.406],
                                      image_std_norm=[0.229, 0.224, 0.225])
    image_dataset = torch.utils.data.ConcatDataset([train_image_dataset, val_image_dataset, test_image_dataset])
    torch.save(image_dataset, "/tmp/image_dataset_rvl.pth")

nlp = fasttext.load_model(conf.core.text_fasttext_model_path)
train_text_dataset = TextDataset('/data01/stefanopio.zingaro/datasets/QS-OCR-Large/train', nlp=nlp)
val_text_dataset = TextDataset('/data01/stefanopio.zingaro/datasets/QS-OCR-Large/val', nlp=nlp)
test_text_dataset = TextDataset('/data01/stefanopio.zingaro/datasets/QS-OCR-Large/test', nlp=nlp)
text_dataset = torch.utils.data.ConcatDataset([train_text_dataset, val_text_dataset, test_text_dataset])

dataset = FusionDataset(image_dataset, text_dataset)
random_dataset_split = torch.utils.data.random_split(dataset, [320000, 40000, 40000])
datasets = OrderedDict({
    'train': random_dataset_split[0],
    'val': random_dataset_split[1],
    'test': random_dataset_split[2]
})
dataloaders = {
    x: DataLoader(datasets[x],
                  batch_size=conf.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'))
    for x in conf.batch_sizes
}

model = FusionSideNetSideFC(300,
                            num_classes=10,
                            alphas=[.2, .4, .4],
                            dropout_prob=.5,
                            side_fc=512).to(conf.device)

_, c = np.unique(np.array(dataset.targets)[dataloaders['train'].dataset.indices], return_counts=True)
weight = torch.from_numpy(np.min(c) / c).type(torch.FloatTensor).to(conf.device)
criterion = nn.CrossEntropyLoss(weight=weight).to(conf.device)
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .1 * (1.0 - float(epoch) / 100.0) ** .5)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
pipeline = TrainingPipeline(model, criterion, optimizer, scheduler, device=conf.device)
best_valid_acc, test_acc, confusion_matrix = pipeline.run(dataloaders['train'],
                                                          dataloaders['val'],
                                                          dataloaders['test'],
                                                          num_epochs=100)
s = f'1280x512x10,sgd,fasttext,min,2-4-4,' \
    f'{best_valid_acc:.3f},' \
    f'{test_acc:.3f},' \
    f'{",".join([f"{r[i] / np.sum(r):.3f}" for i, r in enumerate(confusion_matrix)])}\n'
with open('../test/results.csv', 'a+') as f:
    f.write(s)

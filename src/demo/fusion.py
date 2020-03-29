from __future__ import division, print_function

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
from models import TextImageSideNet, TrainingPipeline, TextImageSideNetBaseFC, TextImageSideNetSideFC

filterwarnings("ignore")
torch.manual_seed(42)
cudnn.deterministic = True

for task in conf.tasks:
    try:
        image_dataset = torch.load(conf.image_dataset_path)
        if task[2] == 'fasttext':
            text_dataset = torch.load(conf.text_dataset_fasttext_path)
            dataset = torch.load(conf.fusion_dataset_fasttext_path)
        else:
            text_dataset = torch.load(conf.text_dataset_custom_path)
            dataset = torch.load(conf.fusion_dataset_custom_path)
    except FileNotFoundError:
        image_dataset = ImageDataset(conf.image_root_dir,
                                     image_width=384,
                                     image_interpolation=Image.BILINEAR,
                                     image_mean_norm=[0.485, 0.456, 0.406],
                                     image_std_norm=[0.229, 0.224, 0.225])
        if task[2] == 'fasttext':
            nlp = fasttext.load_model(conf.core.text_fasttext_model_path)
            text_dataset = TextDataset(conf.text_root_dir, nlp=None)
            dataset = FusionDataset(image_dataset, text_dataset)
            torch.save(dataset, conf.fusion_dataset_fasttext_path)
        else:
            nlp = None
            text_dataset = TextDataset(conf.text_root_dir, nlp=nlp)
            dataset = FusionDataset(image_dataset, text_dataset)
            torch.save(dataset, conf.fusion_dataset_custom_path)
    random_dataset_split = torch.utils.data.random_split(dataset, [800, 200, 2482])
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
    if task[0] == '1280x10':
        model = TextImageSideNet(300,
                                 num_classes=10,
                                 alphas=task[4],
                                 dropout_prob=.5,
                                 custom_embedding=task[2] == 'custom',
                                 custom_num_embeddings=len(text_dataset.lookup)).to(conf.device)
    elif task[0] == '1280x512x10':
        model = TextImageSideNetSideFC(300,
                                       num_classes=10,
                                       alphas=task[4],
                                       dropout_prob=.5,
                                       custom_embedding=task[2] == 'custom',
                                       custom_num_embeddings=len(text_dataset.lookup)).to(conf.device)
    else:
        model = TextImageSideNetBaseFC(300,
                                       num_classes=10,
                                       alphas=task[4],
                                       dropout_prob=.5,
                                       custom_embedding=task[2] == 'custom',
                                       custom_num_embeddings=len(text_dataset.lookup)).to(conf.device)
    if task[3] == 'min':
        _, c = np.unique(np.array(dataset.targets)[dataloaders['train'].dataset.indices], return_counts=True)
        weight = torch.from_numpy(np.min(c) / c).type(torch.FloatTensor).to(conf.device)
    elif task[3] == 'max':
        _, c = np.unique(np.array(dataset.targets)[dataloaders['train'].dataset.indices], return_counts=True)
        weight = torch.from_numpy(np.max(c) / c).type(torch.FloatTensor).to(conf.device)
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight).to(conf.device)
    if task[1] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        scheduler = None
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .1 * (1 - epoch / 100) ** .5)
    pipeline = TrainingPipeline(model, criterion, optimizer, scheduler, device=conf.device)
    best_valid_acc, test_acc, confusion_matrix = pipeline.run(dataloaders['train'],
                                                              dataloaders['val'],
                                                              dataloaders['test'],
                                                              num_epochs=100)
    s = f'{",".join([i for i in task])},' \
        f'{best_valid_acc:.3f},' \
        f'{test_acc:.3f},' \
        f'{",".join([f"{r[i] / np.sum(r):.3f}" for i,r in enumerate(confusion_matrix)])}\n'
    with open('../test/results.csv', 'a+') as f:
        f.write(s)

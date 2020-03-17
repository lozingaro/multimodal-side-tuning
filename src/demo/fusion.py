from __future__ import print_function, division

from warnings import filterwarnings

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

import conf
from datasets.tobacco import TobaccoImageDataset, TobaccoTextDataset
from models.nets import AgneseNetModel

filterwarnings("ignore")

torch.manual_seed(conf.core.seed)
cudnn.deterministic = True

print('\nLoading data...', end=' ')
image_dataset = TobaccoImageDataset(conf.dataset.image_root_dir,
                                    image_width=conf.dataset.image_width,
                                    image_interpolation=conf.dataset.image_interpolation,
                                    image_mean_norm=conf.dataset.image_mean_normalization,
                                    image_std_norm=conf.dataset.image_std_normalization,
                                    splits=conf.dataset.lengths)
text_dataset = TobaccoTextDataset(conf.dataset.text_root_dir,
                                  context=conf.dataset.text_words_per_doc,
                                  num_grams=conf.dataset.text_ngrams,
                                  splits=conf.dataset.lengths,
                                  # fasttext_model_path=conf.dataset.text_fasttext_model_path,
                                  )
fusion_dataloaders = {
    x: DataLoader(torch.utils.data.ConcatDataset([image_dataset.datasets[x], text_dataset.datasets[x]]),
                  batch_size=conf.dataset.batch_sizes[x],
                  shuffle=bool(x == 'train' or x == 'val'),
                  num_workers=0,
                  pin_memory=True)
    for x in conf.dataset.batch_sizes
}
# torch.save(image_dataset, '/tmp/tobacco_image_dataset.pth')
# torch.save(text_dataset, '/tmp/tobacco_text_dataset.pth')
print('done.')

print('\nModel train and evaluation...')
fusion_model = AgneseNetModel().to(conf.core.device)
fusion_criterion = torch.nn.CrossEntropyLoss().to(conf.core.device)
fusion_optimizer = torch.optim.SGD(fusion_model.parameters(), lr=conf.model.fusion_lr, momentum=conf.model.momentum)

for epoch in range(100):

    for phase in ['train', 'val']:

        if phase == 'train':
            fusion_model.train()
        else:
            fusion_model.eval()

        epoch_loss = 0.0
        epoch_acc = 0.0

        for inputs_image, _, inputs_text, labels in fusion_dataloaders[phase]:

            fusion_optimizer.zero_grad()
            inputs_image, inputs_text, labels = inputs_image.to('cuda'), inputs_text.to('cuda'), labels.to('cuda')
            outputs = fusion_model(inputs_image, inputs_text)
            loss = fusion_criterion(outputs, labels)
            epoch_loss += loss.item() * conf.dataset.batch_sizes[phase]

            if phase == 'train':
                loss.backward()
                fusion_optimizer.step()

            _, preds = torch.max(outputs, 1)
            epoch_acc += torch.sum(preds == labels.data)

        print('Epoch: %d' % (epoch + 1))
        print(f'\tLoss: {epoch_loss / len(fusion_dataloaders[phase].dataset):.4f}({phase})\t|'
              f'\tAcc: {epoch_acc / float(len(fusion_dataloaders[phase].dataset)) * 100:.1f}% ({phase})')

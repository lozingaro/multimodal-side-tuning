from __future__ import print_function, division

import copy
import os
import time
import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader
from torchvision import models, transforms, utils

from configs.core import train_batch_size, val_batch_size, test_batch_size, train_length, val_length, test_length, \
    from_scratch
from datasets.tobacco import TobaccoImageDataset, imshow

warnings.filterwarnings("ignore")
plt.ion()  # interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Data
root_dir = '/Users/zingaro/SD128/Developer/multimodal-side-tuning/data/Tobacco3482-jpg'
# root_dir = '/data01/stefanopio.zingaro/datasets/tobacco-image'

image_transforms = {
    # Data augmentation and normalization for training
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Just normalization for validation
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Just normalization for validation, check on the significance of Resize(224)
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_dataset_saved_path = '/tmp/tobacco_image_dataset.pth'
if os.path.exists(image_dataset_saved_path) and not from_scratch:
    image_dataset = torch.load(image_dataset_saved_path)
else:
    image_dataset = TobaccoImageDataset(root_dir, [train_length, val_length, test_length], image_transforms)
    torch.save(image_dataset, image_dataset_saved_path)

image_datasets = {
    'train': image_dataset.train,
    'val': image_dataset.val,
    'test': image_dataset.test,
}
image_dataloaders = {
    'train': DataLoader(image_dataset.train, batch_size=train_batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(image_dataset.val, batch_size=val_batch_size, shuffle=True, num_workers=1),
    'test': DataLoader(image_dataset.test, batch_size=test_batch_size, shuffle=True, num_workers=1),
}

###################################################################################
#                               Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data augmentations.

# Get a batch of training data
inputs, classes = next(iter(image_dataloaders['train']))

# Make a grid from batch
out = utils.make_grid(inputs)
imshow(out, title=[image_dataset.classes[x] for x in classes])


###################################################################################
#                                   Training the model
# ---------------------------------------------------------------------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data_inputs, labels in image_dataloaders[phase]:
                data_inputs = data_inputs.to(torch.device)
                labels = labels.to(torch.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(data_inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * data_inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = float(running_corrects) / len(image_datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure()

    with torch.no_grad():
        for i, (data_inputs, labels) in enumerate(image_dataloaders['test']):
            data_inputs = data_inputs.to(torch.device)
            labels.to(torch.device)

            outputs = model(data_inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(data_inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(image_dataset.classes[preds[j]]))
                imshow(data_inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#

# model_ft = models.resnet18(pretrained=True)
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(image_dataset.classes))

model_ft = model_ft.to(device)

train_criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = Adam(model_ft.parameters(), lr=0.001, weight_decay=0.0)
# optimizer_ft = SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#

model_ft = train_model(model_ft, train_criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

######################################################################

visualize_model(model_ft)

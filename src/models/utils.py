import copy
import math
import time

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

from conf import model, core


def train_image_model(model, dataloaders, criterion, optimizer, scheduler, lengths, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            counter = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if core.use_gpu:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that
                # differentiation can be done automatically.
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    # print('loss done')
                    # Just so that you can keep track that something's happening and don't feel like the program
                    # isn't running.
                    if counter % 10 == 0:
                        print("Reached iteration ", counter)
                    counter += 1

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # print('loss backward')
                        loss.backward()
                        # print('done loss backward')
                        optimizer.step()
                        # print('done optim')

                # print evaluation statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                # print('running correct =', running_corrects)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / lengths[phase]
            epoch_acc = float(running_corrects) / lengths[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def evaluate_model(model, dataloader, length):
    model.eval()
    running_corrects = 0

    for inputs, labels in dataloader:
        if core.use_gpu:
            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    return float(running_corrects) / length


def custom_scheduler(optimizer):
    lr_lambda = lambda epoch: model.image_initial_lr * math.sqrt(1 - epoch / model.image_epochs)
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

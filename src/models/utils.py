import copy

import torch

from conf import core


def train_eval_test(model, dataloaders, optimizer, criterion, scheduler, lengths, num_epochs=25):
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_running_loss, train_running_corrects = _train(model, dataloaders['train'], optimizer, criterion)
        train_epoch_loss = train_running_loss / lengths['train']
        train_epoch_acc = float(train_running_corrects) / lengths['train']
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', train_epoch_loss, train_epoch_acc))

        val_running_loss, val_running_corrects = _eval(model, dataloaders['val'], criterion)
        val_epoch_loss = val_running_loss / lengths['val']
        val_epoch_acc = float(val_running_corrects) / lengths['val']
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', val_epoch_loss, val_epoch_acc))

        scheduler.step()

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model = copy.deepcopy(model.state_dict())

        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model)

    test_running_corrects = _test(model, dataloaders['test'])
    test_acc = float(test_running_corrects) / lengths['test']
    print('{} Acc: {:.4f}'.format('test', test_acc))

    return model


def _train(model, dataloader, optimizer, criterion):
    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(core.device)
        labels = labels.to(core.device)

        # Zero the parameter gradients erasing history
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Print evaluation statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    return running_loss, running_corrects


def _eval(model, dataloader, criterion):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(core.device)
        labels = labels.to(core.device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    return running_loss, running_corrects


def _test(model, dataloader):
    model.eval()

    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(core.device)
        labels = labels.to(core.device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    return running_corrects

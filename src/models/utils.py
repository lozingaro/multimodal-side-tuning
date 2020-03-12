import time
import torch
import conf


class TrainingPipeline:
    def __init__(self, model, optimizer, criterion, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

    def run(self, data_train, data_eval, data_test, num_epochs):
        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss, train_acc = self.train_(data_train)
            valid_loss, valid_acc = self.eval_(data_eval)

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}% (train)')
            print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}% (valid)')

        print('Checking the results of test dataset...')
        test_loss, test_acc = self.eval_(data_test)
        print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}% (test)')

    # def train_eval_test(self, model, dataloaders, optimizer, criterion, scheduler, lengths, num_epochs=25):
    #     best_model = copy.deepcopy(model.state_dict())
    #     best_acc = 0.0
    #
    #     for epoch in range(num_epochs):
    #         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    #         print('-' * 10)
    #
    #         train_running_loss, train_running_corrects = self.train_(model, dataloaders['train'], optimizer, criterion)
    #         train_epoch_loss = train_running_loss / lengths['train']
    #         train_epoch_acc = float(train_running_corrects) / lengths['train']
    #         print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', train_epoch_loss, train_epoch_acc))
    #
    #         val_running_loss, val_running_corrects = self.eval_(model, dataloaders['val'], criterion)
    #         val_epoch_loss = val_running_loss / lengths['val']
    #         val_epoch_acc = float(val_running_corrects) / lengths['val']
    #         print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', val_epoch_loss, val_epoch_acc))
    #
    #         scheduler.step()
    #
    #         if val_epoch_acc > best_acc:
    #             best_acc = val_epoch_acc
    #             best_model = copy.deepcopy(model.state_dict())
    #
    #         print()
    #
    #     print('Best val Acc: {:4f}'.format(best_acc))
    #
    #     # load best model weights
    #     model.load_state_dict(best_model)
    #
    #     test_running_corrects = self.eval_(model, dataloaders['test'])
    #     test_acc = float(test_running_corrects) / lengths['test']
    #     print('{} Acc: {:.4f}'.format('test', test_acc))
    #
    #     return model

    def train_(self, data):
        pass

    def eval_(self, data):
        pass


class ImageTrainingPipeline(TrainingPipeline):
    def __init__(self, model, optimizer, criterion, scheduler):
        super().__init__(model, optimizer, criterion, scheduler)

    def train_(self, data):
        self.model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in data:
            inputs = inputs.to(conf.core.device)
            labels = labels.to(conf.core.device)

            # Zero the parameter gradients erasing history
            self.optimizer.zero_grad()

            # Forward
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Print evaluation statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        return running_loss, running_corrects

    def eval_(self, data):
        self.model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in data:
            inputs = inputs.to(conf.core.device)
            labels = labels.to(conf.core.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        return running_loss, running_corrects


class TextTrainingPipeline(TrainingPipeline):
    def __init__(self, model, optimizer, criterion, scheduler):
        super().__init__(model, optimizer, criterion, scheduler)

    def train_(self, data):
        train_loss = 0
        train_acc = 0

        for i, (text, offsets, label) in enumerate(data):
            self.optimizer.zero_grad()
            text, offsets, label = text.to(conf.core.device), offsets.to(conf.core.device), label.to(conf.core.device)
            output = self.model(text, offsets)
            loss = self.criterion(output, label)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            train_acc += (output.argmax(1) == label).sum().item()

        self.scheduler.step()

        return train_loss / len(data.dataset), train_acc / len(data.dataset)

    def eval_(self, data):
        loss = 0
        acc = 0

        for text, offsets, cls in data:
            text, offsets, cls = text.to(conf.core.device), offsets.to(conf.core.device), cls.to(conf.core.device)
            with torch.no_grad():
                output = self.model(text, offsets)
                loss = self.criterion(output, cls)
                loss += loss.item()
                acc += (output.argmax(1) == cls).sum().item()

        return loss / len(data.dataset), acc / len(data.dataset)

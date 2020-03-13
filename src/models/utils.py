import copy
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
        best_model = copy.deepcopy(self.model.state_dict())
        best_valid_acc = 0.0

        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss, train_acc = self.train_(data_train)
            valid_loss, valid_acc = self.eval_(data_eval)

            if valid_acc > best_valid_acc:
                best_model = copy.deepcopy(self.model.state_dict())

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}% (train)')
            print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}% (valid)')

        self.model.load_state_dict(best_model)  # load best model weights

        print('Checking the results of test dataset...')
        test_loss, test_acc = self.eval_(data_test)
        print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}% (test)')

    def train_(self, data):
        pass

    def eval_(self, data):
        pass


class ImageTrainingPipeline(TrainingPipeline):
    def __init__(self, model, optimizer, criterion, scheduler):
        super().__init__(model, optimizer, criterion, scheduler)

    def train_(self, data):
        self.model.train()

        train_loss = 0.0
        train_acc = 0

        for inputs, labels in data:
            self.optimizer.zero_grad()
            inputs, labels = inputs.to(conf.core.device), labels.to(conf.core.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            train_loss += loss.item() * inputs.size(0)
            loss.backward()
            self.optimizer.step()
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)

        # self.scheduler.step()

        return train_loss / len(data.dataset), train_acc.double() / len(data.dataset)

    def eval_(self, data):
        self.model.eval()

        eval_loss = 0.0
        eval_acc = 0

        for inputs, labels in data:
            inputs, labels = inputs.to(conf.core.device), labels.to(conf.core.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                eval_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                eval_acc += torch.sum(preds == labels.data)

        return eval_loss / len(data.dataset), eval_acc / len(data.dataset)


class TextTrainingPipeline(TrainingPipeline):
    def __init__(self, model, optimizer, criterion, scheduler):
        super().__init__(model, optimizer, criterion, scheduler)

    def train_(self, data):
        self.model.train()

        train_loss = 0
        train_acc = 0

        for text, offsets, label in data:
            self.optimizer.zero_grad()
            text, offsets, label = text.to(conf.core.device), offsets.to(conf.core.device), label.to(conf.core.device)
            output = self.model(text, offsets)
            loss = self.criterion(output, label)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            train_acc += (output.argmax(1) == label).sum().item()

        # self.scheduler.step()

        return train_loss / len(data.dataset), train_acc / len(data.dataset)

    def eval_(self, data):
        self.model.eval()

        eval_loss = 0
        eval_acc = 0

        for text, offsets, label in data:
            text, offsets, label = text.to(conf.core.device), offsets.to(conf.core.device), label.to(conf.core.device)
            with torch.no_grad():
                output = self.model(text, offsets)
                loss = self.criterion(output, label)
                eval_loss += loss.item()
                eval_acc += (output.argmax(1) == label).sum().item()

        return eval_loss / len(data.dataset), eval_acc / len(data.dataset)

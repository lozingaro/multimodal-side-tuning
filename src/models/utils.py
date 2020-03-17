import copy
import time

import torch


class TrainingPipeline:
    def __init__(self, model, optimizer, criterion, scheduler=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

    def run(self, data_train, data_eval=None, data_test=None, num_epochs=5):
        best_model = copy.deepcopy(self.model.state_dict())
        best_valid_acc = 0.0

        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss, train_acc = self.__train(data_train)
            if data_eval is not None:
                valid_loss, valid_acc = self.__eval(data_eval)

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_model = copy.deepcopy(self.model.state_dict())

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}% (train)')
            if data_eval is not None:
                print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}% (valid)')

        self.model.load_state_dict(best_model)  # load best model weights

        if data_test is not None:
            print('Checking the results of test dataset...')
            test_loss, test_acc = self.__eval(data_test)
            print(f'\tBest Acc: {best_valid_acc * 100:.1f}% (valid)')
            print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}% (test)')

    def __train(self, data):
        self.model.train()

        train_loss = 0.0
        train_acc = 0.0

        for inputs, labels in data:
            try:
                self.optimizer.zero_grad()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                train_loss += loss.item() * inputs.size(0)
                loss.backward()
            except:
                print("WTF!")

            self.optimizer.step()
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)


        if self.scheduler is not None:
            self.scheduler.step()

        return train_loss / len(data.dataset), train_acc / float(len(data.dataset))

    def __eval(self, data):
        self.model.eval()

        eval_loss = 0.0
        eval_acc = 0

        for inputs, labels in data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                eval_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                eval_acc += torch.sum(preds == labels.data)

        return eval_loss / len(data.dataset), eval_acc / float(len(data.dataset))


def merge(alpha, base_encoding, side_encoding):
    weights = [alpha, 1 - alpha]
    outputs_to_merge = [base_encoding] + [side_encoding]
    merged_encoding = torch.zeros_like(base_encoding, device=base_encoding.device)
    merged_encoding = merged_encoding

    for a, out in zip(weights, outputs_to_merge):
        merged_encoding += a * out
    return merged_encoding

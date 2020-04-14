import copy
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


class TrainingPipeline:

    def __init__(self, model, criterion, optimizer, scheduler=None,
                 device='cuda', num_classes=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes

    def run(self, data_train, data_eval=None, data_test=None,
            num_epochs=50):
        best_model = copy.deepcopy(self.model.state_dict())
        best_valid_acc = 0.0
        train_distances = []

        try:
            for epoch in range(num_epochs):
                start_time = time.time()
                train_loss, train_acc, epoch_distances = self._train(data_train)
                train_distances += epoch_distances

                valid_loss, valid_acc = .0, .0
                if data_eval is not None:
                    valid_loss, valid_acc, _ = self._eval(data_eval)

                    if valid_acc >= best_valid_acc:
                        best_valid_acc = valid_acc
                        best_model = copy.deepcopy(self.model.state_dict())

                secs = int(time.time() - start_time)
                mins = secs / 60
                secs %= 60

                print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
                print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc:.3f} (train)')
                if data_eval is not None:
                    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc:.3f} (valid)')

        except KeyboardInterrupt:
            pass

        self.model.load_state_dict(best_model)

        test_loss, test_acc, confusion_matrix = 0, 0, None
        if data_test is not None:
            print('Checking the results of test dataset...')
            test_loss, test_acc, confusion_matrix = self._eval(data_test)
            print(f'\tBest Acc: {best_valid_acc:.3f} (valid)')
            print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc:.3f} (test)\n')
            print(f'\n{"Category":10s} - Accuracy')
            for i, r in enumerate(confusion_matrix):
                print(f'{data_test.dataset.classes} - {r[i] / np.sum(r):.3f}')

        plt.plot(train_distances)
        plt.show()
        return best_valid_acc, test_acc, confusion_matrix

    def _train(self, data):
        self.model.train()

        train_loss = 0.0
        train_acc = 0.0
        distances = []

        for _, (inputs, labels) in tqdm(enumerate(data)):
            self.optimizer.zero_grad()
            if type(inputs) is list:
                batch_size = inputs[0].size(0)
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(self.device)
            else:
                batch_size = inputs.size(0)
                inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            result = self.model(inputs)
            if type(result) is tuple:
                outputs = result[0]
                distances.append(result[1])
            else:
                outputs = result
            loss = self.criterion(outputs, labels)
            train_loss += loss.item() * batch_size
            loss.backward()
            self.optimizer.step()
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)

        if self.scheduler is not None:
            self.scheduler.step()

        return train_loss / len(data.dataset), train_acc / float(
            len(data.dataset)), distances

    def _eval(self, data):
        self.model.eval()

        eval_loss = 0.0
        eval_acc = 0
        confusion_matrix = np.zeros([self.num_classes, self.num_classes], int)

        for _, (inputs, labels) in tqdm(enumerate(data)):
            if type(inputs) is list:
                batch_size = inputs[0].size(0)
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(self.device)
            else:
                batch_size = inputs.size(0)
                inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                result = self.model(inputs)
                if type(result) is tuple:
                    outputs = result[0]
                else:
                    outputs = result
                loss = self.criterion(outputs, labels)
                eval_loss += loss.item() * batch_size
                _, preds = torch.max(outputs, 1)
                eval_acc += (preds == labels).sum().item()

                for i, l in enumerate(labels):
                    confusion_matrix[l.item(), preds[i].item()] += 1

        return eval_loss / len(data.dataset), eval_acc / float(
            len(data.dataset)), confusion_matrix


def merge(variables, weights, return_distance=False):
    coeffs = weights + [1 - sum([i for i in weights])]
    res = torch.zeros_like(variables[0], device=variables[0].device)

    for coeff, var in zip(coeffs, variables):
        res += coeff * var

    if return_distance:
        d = [torch.mean(torch.tensor(
            [torch.dist(x[i], y[i]) / len(x[i]) for i in range(len(x))])).item()
             for x, y in [e for e in itertools.combinations(variables, 2)]]
        return res, d
    else:
        return res

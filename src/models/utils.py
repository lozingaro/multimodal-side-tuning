import copy
import time

import torch
import matplotlib.pyplot as plt


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
        train_distances = []

        try:
            for epoch in range(num_epochs):
                start_time = time.time()
                train_loss, train_acc, epoch_distances = self._train(data_train)
                train_distances += epoch_distances

                if data_eval is not None:
                    valid_loss, valid_acc, _, _ = self._eval(data_eval)

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

        except KeyboardInterrupt:
            pass

        self.model.load_state_dict(best_model)  # load best model weights

        if data_test is not None:
            print('Checking the results of test dataset...')
            test_loss, test_acc, class_acc, class_tot = self._eval(data_test)
            print(f'\tBest Acc: {best_valid_acc * 100:.1f}% (valid)')
            print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}% (test)')

            for i in range(10):
                print('Acc of %5s : %2d %%' % (data_test.dataset.dataset.classes[i], 100 * class_acc[i] / class_tot[i]))

        if len(train_distances):
            plt.plot(train_distances)
            plt.show()

    def _train(self, data):
        self.model.train()

        train_loss = 0.0
        train_acc = 0.0

        for inputs, labels in data:
            self.optimizer.zero_grad()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            train_loss += loss.item() * inputs.size(0)
            loss.backward()
            self.optimizer.step()
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)

        if self.scheduler is not None:
            self.scheduler.step()

        return train_loss / len(data.dataset), train_acc / float(len(data.dataset)), []

    def _eval(self, data):
        self.model.eval()

        eval_loss = 0.0
        eval_acc = 0

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        for inputs, labels in data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                eval_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                eval_acc += (preds == labels).sum().item()

                for i in range(4):
                    label = labels[i]
                    class_correct[label] += (preds == labels).squeeze()[i].item()
                    class_total[label] += 1

        return eval_loss / len(data.dataset), eval_acc / float(len(data.dataset)), class_correct, class_total


class FusionTrainingPipeline(TrainingPipeline):
    def __init__(self, model, optimizer, criterion, device):
        super().__init__(model, optimizer, criterion, device=device)

    def _train(self, data):
        self.model.train()

        train_loss = 0.0
        train_acc = 0.0

        distances = []

        for (inputs_image, inputs_text), labels in data:
            self.optimizer.zero_grad()
            inputs_image = inputs_image.to(self.device)
            inputs_text = inputs_text.to(self.device)
            labels = labels.to(self.device)
            outputs, batch_distances = self.model(inputs_image, inputs_text)
            distances.append(torch.mean(batch_distances))
            loss = self.criterion(outputs, labels)
            train_loss += loss.item() * inputs_image.size(0)
            loss.backward()
            self.optimizer.step()
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)

        if self.scheduler is not None:
            self.scheduler.step()

        return train_loss / len(data.dataset), train_acc / float(len(data.dataset)), distances

    def _eval(self, data):
        self.model.eval()

        eval_loss = 0.0
        eval_acc = 0

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        for (inputs_image, inputs_text), labels in data:
            inputs_image = inputs_image.to(self.device)
            inputs_text = inputs_text.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs, _ = self.model(inputs_image, inputs_text)
                loss = self.criterion(outputs, labels)
                eval_loss += loss.item() * inputs_image.size(0)
                _, preds = torch.max(outputs, 1)
                eval_acc += torch.sum(preds == labels.data)
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += (preds == labels).squeeze()[i].item()
                    class_total[label] += 1

        return eval_loss / len(data.dataset), eval_acc / float(len(data.dataset)), class_correct, class_total


def merge(alpha, x, y, return_distance=False):
    if return_distance:
        d = [torch.dist(x[i], y[i]).item()/len(x) for i in range(len(x))]
        return mergeN([alpha], [x, y]), d
    else:
        return mergeN([alpha], [x, y])


def mergeN(ops, x):
    weights = ops + [1 - sum([i for i in ops])]
    out = [i for i in x]
    merged_encoding = torch.zeros_like(x[0], device=x.device)

    for a, out in zip(weights, out):
        merged_encoding += a * out
    return merged_encoding

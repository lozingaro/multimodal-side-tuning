from torch import nn

image_criterion = nn.CrossEntropyLoss()
image_initial_lr = .01
image_momentum = .9
image_epochs = 100


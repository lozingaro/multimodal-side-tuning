import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
alpha = .75

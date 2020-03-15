import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
load_image_dataset = False
load_text_dataset = False
alpha = .5

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
load_image_dataset = True
load_text_dataset = False


import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_root_dir = '/data01/stefanopio.zingaro/datasets/Tobacco3482-jpg'
text_root_dir = '/data01/stefanopio.zingaro/datasets/QS-OCR-small'
batch_sizes = {
    'train': 16,
    'val': 4,
    'test': 32
}
lengths = {
    'train': 800,
    'val': 200,
    'test': 2482
}
text_spacy_model_path = '/data01/stefanopio.zingaro/datasets/spacy/en_vectors_crawl_lg'
text_fasttext_model_path = '/data01/stefanopio.zingaro/datasets/fasttext/cc.en.300.bin'


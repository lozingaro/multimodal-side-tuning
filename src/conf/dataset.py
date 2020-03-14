from PIL import Image

# Image
image_root_dir = '/data01/stefanopio.zingaro/datasets/Tobacco3482-jpg'
batch_sizes = {
    'train': 16,
    'val': 4,
    'test': 128
}
lengths = {
    'train': 800,
    'val': 200,
    'test': 2482
}
image_mean_normalization = [0.485, 0.456, 0.406]
image_std_normalization = [0.229, 0.224, 0.225]
image_interpolation = Image.BILINEAR
image_width = 384

# Text
text_root_dir = '/data01/stefanopio.zingaro/datasets/QS-OCR-small'
text_ngrams = 1
text_vocab_dim = 500
text_embedding_dim = 300

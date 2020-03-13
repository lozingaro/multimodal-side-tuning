from PIL import Image

# Image
image_root_dir = '/data01/stefanopio.zingaro/datasets/Tobacco3482-jpg'
image_batch_sizes = {
    'train': 16,
    'val': 4,
    'test': 128
}
image_lengths = {
    'train': 800,
    'val': 200,
    'test': 2482
}
image_mean_normalization = [0.485, 0.456, 0.406]
image_std_normalization = [0.229, 0.224, 0.225]
image_interpolation = Image.BILINEAR
image_width = 384

# Text
text_root_dir = '/data01/stefanopio.zingaro/datasets/text-classification'
text_ngrams = 2
text_batch_sizes = {
    'train': 16,
    'val': 16,
    'test': 16
}
text_lengths = {
    'train': int(120000 * .95),
    'val': 120000 - int(120000 * .95),
    'test': 7600
}
text_embedding_dim = 32

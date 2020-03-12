from PIL import Image

# Image
image_root_dir = '/data01/stefanopio.zingaro/datasets/Tobacco3482-jpg'
image_batch_sizes = {
    'train': 16,
    'val': 4,
    'test': 128
}
image_train_len = 800
image_val_len = 200
image_test_len = 2482
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
text_train_len = int(120000 * .95)
text_val_len = 120000 - text_train_len
text_test_len = 7600

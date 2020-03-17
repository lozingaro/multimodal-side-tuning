from PIL import Image

image_root_dir = '/data01/stefanopio.zingaro/datasets/Tobacco3482-jpg'
text_root_dir = '/data01/stefanopio.zingaro/datasets/QS-OCR-small'

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

text_words_per_doc = 500
text_embedding_dim = 300
text_spacy_model_path = '/data01/stefanopio.zingaro/datasets/spacy/en_vectors_crawl_lg'
text_fasttext_model_path = '/data01/stefanopio.zingaro/datasets/fasttext/cc.en.300.bin'

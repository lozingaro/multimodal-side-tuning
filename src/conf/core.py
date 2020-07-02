import itertools
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tobacco_img_root_dir = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/Tobacco3482-jpg'
tobacco_txt_root_dir = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/QS-OCR-small'
rlv_img_root_dir = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/RVL-CDIP'
rlv_txt_root_dir = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/QS-OCR-Large'
text_fasttext_model_path = '/home/stefanopio.zingaro/Developer/multimodal-side-tuning/data/cc.en.300.bin'

rvl_labels = ['Letter', 'Form', 'Email', 'Handwritten', 'Advertisement', 'Scientific report', 'Scientific publication', 'Specification', 'File folder', 'News article', 'Budget', 'Invoice', 'Presentation', 'Questionnaire', 'Resume', 'Memo']
tobacco_labels = ['Advertisement', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']

tasks_classifier = [
    # '1280x128x10',
    # '1280x256x10',
    # '1280x512x10',
    'resnet',
    'mobilenet',
    'resnet-direct',
    'mobilenet-direct'
]
tasks_optimizer = ['sgd']  # ['sgd', 'adam']
tasks_embedding = ['fasttext']  # ['fasttext', 'custom']
tasks_loss_weigth = [
    'no',
    # 'min',
    # 'max'
]

tasks_coeffs = ['2-3-5',
                '2-4-4',
                '2-5-3',
                '3-2-5',
                '3-3-4',
                '3-4-3',
                '3-5-2',
                '4-2-4',
                '4-3-3',
                '4-4-2',
                '5-2-3',
                '5-3-2']
tasks = itertools.product(tasks_classifier, tasks_optimizer, tasks_embedding, tasks_loss_weigth, tasks_coeffs)

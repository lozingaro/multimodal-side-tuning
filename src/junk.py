
# Select the dataset image_dataset = datasets.ImageFolder(root_dir) partial_sums = np.unique(image_dataset.targets,
# return_counts=True)[1] partial_probs = [x / len(image_dataset) for x in partial_sums]  # check the distribution
# before sampling random_dataset_split = random_split(image_dataset, [train_length, val_length, test_length])
# datasets = { 'train': random_dataset_split[0], 'val': random_dataset_split[1], 'test': random_dataset_split[2],
# } partial_sums_train = np.unique([image_dataset.targets[i] for i in datasets['train'].indices],
# return_counts=True)[1] partial_probs_train = [x / len(datasets['train']) for x in partial_sums_train]  # check the
# distribution after sampling
#
# # Apply image transformation if needed
# data_transforms = {
#     # Data augmentation and normalization for training
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     # Just normalization for validation
#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     # Just normalization for validation, check on the significance of Resize(224)
#     'test': transforms.Compose([
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
# dataloaders = {
#     'train': DataLoader(datasets['train'], batch_size=train_batch_size, shuffle=True, num_workers=4),
#     'val': DataLoader(datasets['val'], batch_size=val_batch_size, shuffle=True, num_workers=1),
#     'test': DataLoader(datasets['test'], batch_size=test_batch_size, shuffle=True, num_workers=1),
# }
# dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}
# class_names = image_dataset.classes

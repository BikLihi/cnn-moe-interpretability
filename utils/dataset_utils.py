import os
import random
import shutil
import numpy as np

import torchvision
import torch

from copy import copy

import cv2


def train_test_split(dataset, proportions, seed=None):
    # Setting seeds
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    lengths = []
    for prop in proportions:
        lengths.append(int(len(dataset) * prop))

    # Adjusting last length to meet exact length of dataset
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    subsets = torch.utils.data.random_split(dataset, lengths)
    for subset in subsets:
        subset.targets = np.array(subset.dataset.targets)[
            subset.indices].tolist()
        subset.labels = dataset.labels
        subset.dataset = copy(dataset)
    return subsets


def build_subset(dataset, labels):
    if not set(map(str, labels)).issubset(set(map(str, dataset.labels))):
        raise RuntimeError(
            'Specified labels are not a subset of the dataset\' labels')
    mask = np.array([target in labels for target in dataset.targets])
    indices = np.where(mask)[0]
    subset = torch.utils.data.Subset(dataset, indices)
    subset.targets = np.array(dataset.targets)[mask].tolist()
    subset.transform = dataset.transform
    subset.labels = labels
    return subset




def get_transformation(dataset, phase='training'):
    if dataset == 'cifar100':
        if phase == 'training':
            return torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                torchvision.transforms.RandomCrop(size=32, padding=4),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023])
            ])

        else:
            return torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023])
            ])

    if dataset == 'mnist':
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.1307, std=0.3015)
        ])

    if dataset == 'swiss_mnist':
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: 1 - x),
            torchvision.transforms.Normalize(mean=0.0349, std=0.0954),
        ])
    
    if dataset == 'no_transform':
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])


    raise Exception('Dataset or phase not found')

def create_testset(training_dir, labels, ratio, create_dirs=False):
    random.seed(0)

    for label in labels:#
        if create_dirs:
            try:
                os.mkdir(os.path.join(training_dir.replace('training', 'testing'), str(label)))
            except OSError:
                print('Creaton of directory %s failed', os.path.join(training_dir.replace('training', 'testing'), str(label)))

        paths = []
        for file in os.listdir(os.path.join(training_dir, str(label))):
            file_path = os.path.join(training_dir, str(label), file)
            paths.append(file_path)
        samples = random.sample(paths, k=round(len(paths) * ratio))
        for sample in samples:
            shutil.move(sample, sample.replace('training', 'testing'))


def calculate_statistics(dataset, channels=1, batch_size=256):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    mean = 0.0
    std = 0.0
    nb_samples = 0.0
    for data in dataloader:
        data = data[:][0]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    
    mean = mean / nb_samples
    std = std / nb_samples
    
    return mean, std


def get_coco_augmentations():
    augmentation = {
        'RANDRESIZE': True, # enable random resizing
        'JITTER': 0.3, # amplitude of jitter for resizing
        'RANDOM_PLACING': True, # enable random placing
        'HUE': 0.1, # random distortion parameter
        'SATURATION': 1.5, # random distortion parameter
        'EXPOSURE': 1.5, # random distortion parameter
        'LRFLIP': True, # enable horizontal flip
        'RANDOM_DISTORT': False, # enable random distortion in HSV space
    }
    return augmentation


def load_retinanet_image(image_path):
    # load images
    image = cv2.imread(image_path)
    
    if image is None:
        return None

    rows, cols, cns = image.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    return image
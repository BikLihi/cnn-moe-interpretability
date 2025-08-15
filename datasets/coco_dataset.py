import os
import numpy as np

import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO

from models.yolo.utils.utils import *


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(self, mode, classes=None, data_dir='/home/lb4653/mixture-of-experts-thesis/data/coco/images/',
                 ann_dir='/home/lb4653/mixture-of-experts-thesis/data/coco/annotations',
                 img_size=416, min_size=1, debug=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        assert mode in ['train', 'val']

        self.root = data_dir + '{}2017'.format(mode)
        self.annFile = ann_dir + '/instances_{}2017.json'.format(mode)
        if mode == 'train':
            augmentation = {
                'RANDRESIZE': True,  # enable random resizing
                'JITTER': 0.3,  # amplitude of jitter for resizing
                'RANDOM_PLACING': True,  # enable random placing
                'HUE': 0.1,  # random distortion parameter
                'SATURATION': 1.5,  # random distortion parameter
                'EXPOSURE': 1.5,  # random distortion parameter
                'LRFLIP': True,  # enable horizontal flip
                'RANDOM_DISTORT': False,  # enable random distortion in HSV space
            }
        elif mode == 'val':
            augmentation = {'LRFLIP': False, 'JITTER': 0, 'RANDOM_PLACING': False,
                            'HUE': 0, 'SATURATION': 0, 'EXPOSURE': 0, 'RANDOM_DISTORT': False}

        self.mode = mode
        self.classes = classes
        self.data_dir = data_dir + '{}2017'.format(mode)
        self.coco = COCO(self.annFile)
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.lrflip = augmentation['LRFLIP']
        self.jitter = augmentation['JITTER']
        self.random_placing = augmentation['RANDOM_PLACING']
        self.hue = augmentation['HUE']
        self.saturation = augmentation['SATURATION']
        self.exposure = augmentation['EXPOSURE']
        self.random_distort = augmentation['RANDOM_DISTORT']

        if classes:
            catIds = []
            imgIds = set()
            # Iterate over indivual classes
            for class_name in classes:
                catIds_current = self.coco.getCatIds(catNms=[class_name])
                # Assert that cadIds is not empty (invalid class name)
                assert catIds_current, str(class_name) + ' not in coco classes'
                catIds += catIds_current
                imgIds.update(self.coco.getImgIds(catIds=catIds_current))
            self.class_ids = catIds
            self.ids = sorted(list(imgIds))

        else:
            self.class_ids = self.coco.getCatIds()
            self.ids = self.coco.getImgIds()

        if debug:
            self.ids = self.ids[1:2]
            print("debug mode...", self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        annotations = [
            t for t in annotations if t['category_id'] in self.class_ids]

        lrflip = False
        if np.random.rand() > 0.5 and self.lrflip == True:
            lrflip = True

        # load image and preprocess
        img_file = os.path.join(self.root, '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)

        # if 'instances_val5k.json' in self.annFile and img is None:
        #     img_file = os.path.join(self.data_dir, 'train2017',
        #                             '{:012}'.format(id_) + '.jpg')
        #     img = cv2.imread(img_file)
        assert img is not None

        img, info_img = preprocess(img, self.img_size, jitter=self.jitter,
                                   random_placing=self.random_placing)

        if self.random_distort:
            img = random_distort(img, self.hue, self.saturation, self.exposure)

        img = np.transpose(img / 255., (2, 0, 1))

        if lrflip:
            img = np.flip(img, axis=2).copy()

        # load labels
        labels = []
        for anno in annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                labels.append([])
                labels[-1].append(self.class_ids.index(anno['category_id']))
                labels[-1].extend(anno['bbox'])

        padded_labels = np.zeros((self.max_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            labels = label2yolobox(labels, info_img, self.img_size, lrflip)
            padded_labels[range(len(labels))[:self.max_labels]
                          ] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)

        return img, padded_labels, info_img, id_

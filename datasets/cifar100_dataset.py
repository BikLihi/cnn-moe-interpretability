from torch.utils.data import Subset
import numpy as np
from datasets.base_dataset import BaseDataset

from utils.cifar100_utils import CIFAR100_SUPERCLASSES, CIFAR100_SUPERCLASS_ENCODING, CIFAR100_ENCODING, get_superclass


class CIFAR100Dataset(BaseDataset):

    def __init__(self, root_dir, labels=None, superclass_labels=False, transform=None, target_transform=None, label_encoding=True):
        self.superclass_labels = superclass_labels
       
        if superclass_labels:
            subclass_labels = []
            for cls in labels:
                subclass_labels += list(CIFAR100_SUPERCLASSES[cls])
            super().__init__(root_dir, subclass_labels, transform)
        
        else:
            super().__init__(root_dir, labels, transform, target_transform, label_encoding, CIFAR100_ENCODING)

        if superclass_labels:
            self.targets = list(map(get_superclass, self.targets))
            self.labels = list(set(map(get_superclass, self.labels)))
        # if label_encoding:
        #     if superclass_labels:
        #         self.targets = list(
        #             map(CIFAR100_SUPERCLASS_ENCODING.get, self.targets))
        #     else:
        #         self.targets = list(map(CIFAR100_ENCODING.get, self.targets))


    def build_subset(self, labels):
        if self.encode_targets:
            labels = list(map(CIFAR100_ENCODING.get, labels))

        if not set(labels).issubset(self.labels):
            raise RuntimeError(
                'Specified labels are not a subset of the dataset\' labels')
        mask = np.array([target in labels for target in self.targets])
        indices = np.where(mask)[0]
        subset = Subset(self, indices)
        subset.targets = np.array(self.targets)[mask]
        return subset
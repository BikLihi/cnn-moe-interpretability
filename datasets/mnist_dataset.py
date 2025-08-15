from datasets.base_dataset import BaseDataset


class MNISTDataset(BaseDataset):
    def __init__(self, root_dir, labels=None, transform=None, target_transform=None, label_encoding=False):
        super().__init__(root_dir, labels, transform, target_transform, label_encoding)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        return sample, int(target)
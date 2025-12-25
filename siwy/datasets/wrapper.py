import torch
from torch.utils.data import Dataset


class LabelToIdxWrapper(Dataset):
    def __init__(self, base_ds, class_to_idx, max_length=5, transform=None):
        self.base_ds = base_ds
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.len = max_length

    def __len__(self):
        # TODO: setup max length for testing as parameter
        # TODO: explain it in readme
        return self.len

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        if isinstance(label, str):
            label = self.class_to_idx[label]
        return img, torch.tensor(label, dtype=torch.long)

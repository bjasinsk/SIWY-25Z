import torch
from torch.utils.data import Dataset


class LabelToIdxWrapper(Dataset):
    def __init__(self, base_ds, class_to_idx, transform=None):
        self.base_ds = base_ds
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return 5

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        if isinstance(label, str):
            label = self.class_to_idx[label]
        return img, torch.tensor(label, dtype=torch.long)

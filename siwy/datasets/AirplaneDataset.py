import pathlib

from torch.utils.data import Dataset

CLASS_NAME = "Airplane"


class AirplaneDataset(Dataset):
    def __init__(self, root: pathlib.Path, transform=None):
        self.img_dir = root
        self.transform = transform
        self.img_labels = list(self.img_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.img_labels)

    @property
    def classes(self):
        return [CLASS_NAME]

    def __getitem__(self, idx):
        img = self.img_labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, CLASS_NAME

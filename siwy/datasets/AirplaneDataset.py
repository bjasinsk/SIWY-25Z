import pathlib

from PIL import Image
from torch.utils.data import Dataset

CLASS_NAME = "Airplane"


class AirplaneDataset(Dataset):
    ARTIFACT_NAME = "airplanes"

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
        img_path = self.img_labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, CLASS_NAME

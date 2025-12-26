import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from siwy.common import MEAN, STD
from siwy.config import PROCESSED_DATA_DIR
from siwy.datasets import CatDogConfig
from siwy.datasets.AirplaneDataset import AirplaneDataset as AirplaneDatasetClass

TORCH_DATASETS = {
    "ImageFolder": ImageFolder,
    "Airplane": AirplaneDatasetClass,
}

DATASETS = [
    "bus-and-truck-easy-val",
    "bus-and-truck-easy-train",
    AirplaneDatasetClass.ARTIFACT_NAME,
    CatDogConfig.ARTIFACT_NAME,
    "bus-and-truck-difficult-val",
    "bus-and-truck-difficult-train",
]

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


def load_dataset(dataset_name: str):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} is not recognized. Available datasets: {DATASETS}")

    return torch.load(PROCESSED_DATA_DIR / f"{dataset_name}.pt", weights_only=False)

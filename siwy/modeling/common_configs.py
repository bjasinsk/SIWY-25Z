"""
Lightweight registry of dataset configurations for unified evaluation and training scripts.
Imports from existing dataset configs to avoid duplication.
"""

from siwy.datasets.BusTruckConfig import (
    BUS_AND_TRUCK_MODEL_ARTIFACT_TEMPLATE,
)
from siwy.datasets.BusTruckConfig import (
    CLASS_TO_IDX as BUS_TRUCK_CLASS_TO_IDX,
)
from siwy.datasets.CatDogConfig import (
    CAT_AND_DOG_MODEL_ARTIFACT_TEMPLATE,
)
from siwy.datasets.CatDogConfig import (
    CLASS_TO_IDX as CAT_DOG_CLASS_TO_IDX,
)
from siwy.datasets.HorseElephantConfig import (
    CLASS_TO_IDX as HORSE_ELEPHANT_CLASS_TO_IDX,
)
from siwy.datasets.HorseElephantConfig import (
    HORSE_AND_ELEPHANT_MODEL_ARTIFACT_TEMPLATE,
)

# Simple lookup - just what varies between scripts
DATASET_CONFIGS = {
    "dog-and-cat": {
        "artifact_template": CAT_AND_DOG_MODEL_ARTIFACT_TEMPLATE,
        "class_to_idx": CAT_DOG_CLASS_TO_IDX,
        "default_epochs": [0, 1, 2, 4, 6, 8],
        "train_split": "train",
        "num_classes": 3,
    },
    "bus-and-truck-easy-train": {
        "artifact_template": BUS_AND_TRUCK_MODEL_ARTIFACT_TEMPLATE,
        "class_to_idx": BUS_TRUCK_CLASS_TO_IDX,
        "default_epochs": [0, 2],
        "train_split": None,
        "num_classes": 3,
    },
    "horse-and-elephant-easy-train": {
        "artifact_template": HORSE_AND_ELEPHANT_MODEL_ARTIFACT_TEMPLATE,
        "class_to_idx": HORSE_ELEPHANT_CLASS_TO_IDX,
        "default_epochs": [0, 1, 2, 4, 6, 8],
        "train_split": None,
        "num_classes": 3,
    },
}

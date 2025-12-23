from loguru import logger
import torch
from torchvision.models import ResNet18_Weights, resnet18

"""
Resnet9 model definition
Based on https://trak.readthedocs.io/en/latest/quickstart.html#id1:~:text=Training%20code%20for%20CIFAR%2D10
"""


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def construct_rn9(num_classes: int):
    assert num_classes is not None and isinstance(num_classes, int) and num_classes > 0, (
        "num_classes must be positive integer for resnet9 "
    )

    def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(inplace=True),
        )

    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2),
    )
    return model


def construct_rn18(num_classes: int, weights=ResNet18_Weights.DEFAULT):
    assert num_classes is not None and isinstance(num_classes, int) and num_classes > 0, (
        "num_classes must be positive integer for resnet18"
    )
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    logger.debug(model)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    logger.debug(model)
    return model


MODELS = {"resnet18-pretrained": construct_rn18, "resnet9": construct_rn9}

if __name__ == "__main__":
    construct_rn18(10)
    construct_rn18(9)

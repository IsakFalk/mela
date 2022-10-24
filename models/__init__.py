from .convnet import convnet4
from .resfc import resfc
from .resnet import resnet12, resnet18, seresnet12, seresnet18
from .resnet_new import resnet50
from .wresnet import wrn_28_10

model_pool = [
    "convnet4",
    "resnet12",
    "seresnet12",
    "wrn_28_10",
]

model_dict = {
    "wrn_28_10": wrn_28_10,
    "convnet4": convnet4,
    "resnet12": resnet12,
    "resnet18": resnet18,
    "seresnet12": seresnet12,
    "seresnet18": seresnet18,
    "resnet50": resnet50,
    "resfc": resfc,
}

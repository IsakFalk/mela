from __future__ import print_function

import torch

from . import model_dict


def create_model(name, dataset="miniImageNet"):
    """create model by name"""
    if dataset.endswith("ImageNet") or dataset.startswith("mix"):
        # Not sure what vX means
        if name.endswith("v2") or name.endswith("v3"):
            model = model_dict[name]()
        elif name.startswith("resnet50") or name.startswith("wrn") or name.startswith("convnet"):
            model = model_dict[name]()
        elif name.startswith("resnet") or name.startswith("seresnet"):
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=5)
        else:
            raise NotImplementedError("model {} not supported in dataset {}:".format(name, dataset))
    elif dataset.lower() == "timit":
        if name.lower() == "fc_res" or name.lower() == "resfc":
            model = model_dict["resfc"]()
    elif "mixed" in dataset.lower() or "h_aircraft" in dataset.lower() or "metadataset" in dataset.lower():
        if name.endswith("v2") or name.endswith("v3"):
            model = model_dict[name]()
        elif name.startswith("resnet50") or name.startswith("wrn") or name.startswith("convnet"):
            model = model_dict[name]()
        elif name.startswith("resnet") or name.startswith("seresnet"):
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=5)
        elif name.lower() == "fc_res" or name.lower() == "resfc":
            model = model_dict["resfc"]()
        else:
            raise NotImplementedError("model {} not supported in dataset {}:".format(name, dataset))
    elif dataset == "CIFAR-FS" or dataset == "FC100":
        if name.startswith("resnet") or name.startswith("seresnet"):
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=2)
        elif name.startswith("convnet"):
            model = model_dict[name]()
        else:
            raise NotImplementedError("model {} not supported in dataset {}:".format(name, dataset))
    else:
        raise NotImplementedError("dataset not supported: {}".format(dataset))

    return model


def get_teacher_name(model_path):
    """parse to get teacher model name"""
    segments = model_path.split("/")[-2].split("_")
    if ":" in segments[0]:
        return segments[0].split(":")[-1]
    else:
        if segments[0] != "wrn":
            return segments[0]
        else:
            return segments[0] + "_" + segments[1] + "_" + segments[2]


def load_teacher(model_path, n_cls, dataset="miniImageNet"):
    """load the teacher model"""
    print("==> loading teacher model")
    model_t = get_teacher_name(model_path)
    model = create_model(model_t, n_cls, dataset)
    model.load_state_dict(torch.load(model_path)["model"])
    print("==> done")
    return model

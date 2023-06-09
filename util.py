import logging
import os
import sys
from collections.abc import Iterable
from copy import copy

import numpy as np
import torch
from omegaconf import open_dict

BATCH_SIZE_FACTOR = {
    "ilsvrc_2012": 7,
    "imagenet": 7,
    "omniglot": 1,
    "aircraft": 1,
    "cu_birds": 1,
    "dtd": 1,
    "quickdraw": 1,
    "fungi": 1,
    "vgg_flower": 1
}

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, Iterable):
            val = [val]

        val = np.asarray(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_up_cudnn():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.determnistic = False


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        idx = torch.from_numpy(np.asarray(topk) - 1)
        return correct.cumsum(0).sum(1)[idx] * 100.0 / batch_size


def to_cuda_list(obj, device=0):
    if torch.cuda.is_available():
        if isinstance(obj, Iterable):
            return [to_cuda_maybe(e) for e in obj]
        else:
            return to_cuda_maybe(obj)
    return obj


def to_cuda_maybe(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cuda()
    return obj


def cuda_to_np(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(np.float64)
    return arr.astype(np.float64)


def print_metrics(names, vals):
    ret = ""
    for key, val in zip(names, vals):
        ret += f"{key}: {val}  "
    return ret


def np_to_cuda(arr, device=0):
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr.astype(np.float32)).cuda(device)

    return arr


def partial_reload(model, state_dict):
    cur_dict = model.state_dict()
    partial_dict = {}
    for k, v in state_dict.items():
        if k in cur_dict and cur_dict[k].shape == v.shape:
            partial_dict[k] = v
    print(f"number of matched tensors: {len(partial_dict)}")
    print(partial_dict.keys())
    cur_dict.update(partial_dict)
    model.load_state_dict(cur_dict)


def change_param_prefix(params, old_prefix, new_prefix):
    _len = len(old_prefix)
    names = list(params.keys())
    for name in names:
        if name.startswith(old_prefix):
            new_name = f"{new_prefix}{name[_len:]}"
            params[new_name] = params[name]
            del params[name]

    return params


def map_dataset_to_config_form(dataset: str) -> str:
    """Map from datasets to forms made for the loading"""
    if dataset.lower() == "mixed":
        datasets = "aircraft/aircraft,cu_birds/cub,vgg_flower/vgg"
    elif dataset.lower() == "metadataset":
        datasets = "aircraft,cub,dtd,fungi,imagenet,omniglot,quickdraw,vgg,mscoco,traffic_sign"
    elif dataset.lower() == "h_aircraft":
        datasets = "aircraft/aircraft_variant,aircraft/aircraft_maker,aircraft/aircraft_family"
    elif dataset.lower() == "miniimagenet":
        datasets = "miniimagenet"
    elif dataset.lower() == "tieredimagenet":
        datasets = "tieredimagenet"
    elif dataset.lower() == "mini60":
        datasets = "x60ImageNet"
    elif dataset.lower() == "tiered780":
        datasets = "x780ImageNet"
    else:
        raise Exception("No such dataset")

    return datasets


def get_logger(name, dir="logs/", file_name=None, log_level=logging.INFO):
    logger = logging.getLogger(name)

    c_handler = logging.StreamHandler(stream=sys.stdout)
    c_handler.setLevel(log_level)
    c_format = logging.Formatter("%(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    file_path = os.path.join(dir, file_name)
    f_handler = logging.FileHandler(file_path)
    # f_handler.setLevel(log_level)
    f_format = logging.Formatter("%(asctime)s | %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    logger.setLevel(log_level)

    return logger


def change_opt(opt, change_dict):
    opt_new = copy(opt)
    with open_dict(opt_new):
        for key, val in change_dict.items():
            opt_new[key] = val
    return opt_new


def save_routine(epoch, model, optimizer, save_path, opt=None):
    state = {"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict(), "opt": opt}
    torch.save(state, save_path)

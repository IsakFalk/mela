import hydra
import torch
import numpy as np
from omegaconf import open_dict
from torch import optim as optim

import util
from util import BATCH_SIZE_FACTOR
from dataset.data_util import get_datasets_mdl, get_datasets
from dataset.mix_dataset import MixDataset
from models.util import create_model
from routines import parse_option
from train_routine import (
    MDLClassifier,
    full_train_mdl,
    get_dataloaders,
)

def create_mask_fn(ds_dict):
    names = ds_dict.keys()
    mask_array = []
    for name in names:
        mask_array.extend([name] * ds_dict[name]["n_cls"])
    mask_array = np.array(mask_array)
    return lambda x: mask_array == x


@hydra.main(config_path="config", config_name="sup_baseline.yaml")
def sup_baseline_main(opt):
    opt = parse_option(opt)
    with open_dict(opt):
        opt.model_name = f"{opt.model_name}_sup_base_best"

    logger = util.get_logger(opt.logger_name, file_name=f"{opt.logger_name}_{opt.model_name}")
    logger.info(opt)

    # Get dictionary of dataset names and number of classes for each dataset
    ds_dict = get_datasets_mdl(opt, "train", opt.rotate_aug)
    # Create mask function which maps from dataset name to boolean array
    mask_fn = create_mask_fn(ds_dict)

    # Create dataloader for each of these taking into account the batch size factor
    train_datasets = [ds_dict[name]["dataset"] for name in ds_dict.keys()]
    train_batch_sizes = [opt.batch_size * BATCH_SIZE_FACTOR[name] for name in ds_dict.keys()]
    trainloaders = get_dataloaders(train_datasets, train_batch_sizes, opt.num_workers)
    class AutoIterLoader:
        def __init__(self, loader):
            self.loader = loader
            self.iterator = iter(self.loader)

        def sample(self):
            try:
                return next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
            return next(self.iterator)

    trainloaders = {
        name: AutoIterLoader(trainloaders[i]) for i, name in enumerate(ds_dict.keys())
    }
    dataloader_info_object = {
        name: {} for name in ds_dict.keys()
    }
    for name in ds_dict.keys():
        dataloader_info_object[name]["n_cls"] = ds_dict[name]["n_cls"]
    aux_info_object = {
        "mask_fn": mask_fn,
        "opt": opt
    }

    # Val datasets stay the same
    val_datasets, _ = get_datasets(opt, "val", False)
    valloaders = get_dataloaders(val_datasets, 256, opt.num_workers, shuffle=False)

    backbone = create_model(opt.model, dataset=opt.dataset)

    model = MDLClassifier(backbone, opt.feat_dim, dataloader_info_object, aux_info_object)
    model = model.cuda()

    if opt.SGD:
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt.learning_rate,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
        )
    else:
        optimizer = optim.Adam(model.parameters())

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_decay_epochs, gamma=opt.lr_decay_rate)

    full_train_mdl(
        opt,
        model,
        trainloaders,
        valloaders,
        optimizer,
        scheduler,
        logger,
        lambda x: x >= opt.lr_decay_epochs[0],
    )


if __name__ == "__main__":
    sup_baseline_main()

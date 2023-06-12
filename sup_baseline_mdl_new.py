#!/usr/bin/env python3

import hydra
import torch
import numpy as np
from omegaconf import open_dict
from torch import optim as optim

import util
from util import BATCH_SIZE_FACTOR
from dataset.data_util import get_datasets_mdl, get_datasets
from dataset.mix_dataset import MixDataset, MDLMixDataset
from models.util import create_model
from routines import parse_option
from train_routine import (
    MDLClassifier,
    full_train_mdl,
    get_dataloaders,
)

@hydra.main(config_path="config", config_name="sup_baseline.yaml")
def sup_baseline_main(opt):
    opt = parse_option(opt)
    with open_dict(opt):
        opt.model_name = f"{opt.model_name}_sup_base_best"

    logger = util.get_logger(opt.logger_name, file_name=f"{opt.logger_name}_{opt.model_name}")
    logger.info(opt)

    ds_dict = get_datasets_mdl(opt, "train", opt.rotate_aug)
    train_datasets = [ds_dict[ds_name] for ds_name in opt.train_datasets]
    batch_size_factor = [BATCH_SIZE_FACTOR[ds_name] for ds_name in opt.train_datasets]
    traindataset = MDLMixDataset(train_datasets, batch_size_factor)
    trainloader = get_dataloaders([traindataset], 128, opt.num_workers)

    # Val datasets stay the same
    val_datasets, _ = get_datasets(opt, "val", False)
    valloaders = get_dataloaders(val_datasets, 256, opt.num_workers, shuffle=False)

    backbone = create_model(opt.model, dataset=opt.dataset)

    model = MDLClassifier(backbone, opt.feat_dim) #, dataloader_info_object, aux_info_object)
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

    full_train_mdl_new(
        opt,
        model,
        trainloader,
        valloaders,
        optimizer,
        scheduler,
        logger,
        lambda x: x >= opt.lr_decay_epochs[0],
    )


if __name__ == "__main__":
    sup_baseline_main()

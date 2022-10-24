import hydra
import torch
from omegaconf import open_dict
from torch import optim as optim

import util
from dataset.data_util import get_datasets, get_meta_dataset
from meta_learner import MetaLS
from models.util import create_model
from routines import parse_option
from train_routine import full_train, get_dataloaders


@hydra.main(config_path="config", config_name="learn_meta_repr.yaml")
def learn_meta_repr(opt):
    opt = parse_option(opt)
    with open_dict(opt):
        opt.model_name = f"{opt.model_name}_meta_repr_best"

    logger = util.get_logger(opt.logger_name, file_name=f"{opt.logger_name}_{opt.model_name}")
    logger.info(opt)

    train_datasets, _ = get_datasets(opt, "train", opt.rotate_aug)
    meta_train_dataset = get_meta_dataset(opt, train_datasets)
    # NOTE: we only want one dataloader so we just pass a singleton list
    meta_trainloader = get_dataloaders([meta_train_dataset], 1, opt.num_workers)[0]

    val_datasets, _ = get_datasets(opt, "val", opt.rotate_aug)
    valloaders = get_dataloaders(val_datasets, 256, opt.num_workers, shuffle=False)

    backbone = create_model(opt.model, dataset=opt.dataset)
    model = MetaLS(backbone, opt, opt.feat_dim)
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

    full_train(
        opt,
        model,
        meta_trainloader,
        valloaders,
        optimizer,
        scheduler,
        logger,
        lambda x: x >= opt.lr_decay_epochs[0],
    )


if __name__ == "__main__":
    learn_meta_repr()

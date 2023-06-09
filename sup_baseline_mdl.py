import hydra
import torch
from omegaconf import open_dict
from torch import optim as optim

import util
from dataset.aircraft import AirCraftMultiHeadSupDataset
from dataset.data_util import get_datasets_mdl
from dataset.mix_dataset import MixDataset
from models.util import create_model
from routines import parse_option
from train_routine import (
    Classifier,
    SemanticMultiheadClassifier,
    full_train,
    get_dataloaders,
)

BATCH_SIZE_FACTOR = {
    "ilsvrc_2012": 7,
    "omniglot": 1,
    "aircraft": 1,
    "cu_birds": 1,
    "dtd": 1,
    "quickdraw": 1,
    "fungi": 1,
    "vgg_flower": 1
}

def get_train_dataset_and_loaders(opt):
    if opt.dataset.lower() == "h_aircraft":
        dataset = AirCraftMultiHeadSupDataset(
            opt, partition="train", rotate_aug=False  # Note possible to augment using rotation
        )
        n_cls = dataset.n_cls
        trainloader = get_dataloaders([dataset], opt.batch_size, opt.num_workers)[0]
    elif opt.dataset.lower() == "mixed":
        datasets, n_cls = get_datasets(opt, "train", opt.rotate_aug)
        dataset = MixDataset(
            datasets, opt, partition="train", rotate_aug=opt.rotate_aug
        )  # NOTE: partition here does nothing
        trainloader = get_dataloaders([dataset], opt.batch_size, opt.num_workers)[0]
    elif opt.dataset.lower() == "metadataset":

    else:
        # Single domain datasets:
        # mini, mini60, tiered, tiered60
        dataset, n_cls = get_datasets(opt, "train", opt.rotate_aug)
        trainloader = get_dataloaders(dataset, opt.batch_size, opt.num_workers)[0]
    return trainloader, n_cls


@hydra.main(config_path="config", config_name="sup_baseline.yaml")
def sup_baseline_main(opt):
    opt = parse_option(opt)
    with open_dict(opt):
        opt.model_name = f"{opt.model_name}_sup_base_best"

    print(opt.datasets)

    logger = util.get_logger(opt.logger_name, file_name=f"{opt.logger_name}_{opt.model_name}")
    logger.info(opt)

    trainloader, n_cls = get_train_dataset_and_loaders(opt)

    val_datasets, _ = get_datasets(opt, "val", False)
    valloaders = get_dataloaders(val_datasets, 256, opt.num_workers, shuffle=False)

    backbone = create_model(opt.model, dataset=opt.dataset)

    if opt.rotate_aug:
        n_cls *= 4

    if opt.dataset.lower() == "h_aircraft":
        model = SemanticMultiheadClassifier(backbone, opt.feat_dim, n_cls)
    else:
        model = Classifier(backbone, opt.feat_dim, n_cls)
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
        trainloader,
        valloaders,
        optimizer,
        scheduler,
        logger,
        lambda x: True,
    )


if __name__ == "__main__":
    sup_baseline_main()

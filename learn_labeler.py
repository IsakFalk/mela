import hydra
import torch
from omegaconf import open_dict
from torch import optim as optim

import util
from dataset.base_dataset import LabelerDataset
from dataset.data_util import get_datasets, get_meta_dataset, get_transform
from models.util import create_model
from routines import get_labeler, get_labeler_accuracy, label_dataset, parse_option
from train_routine import Classifier, full_train, get_dataloaders
from util import change_opt


@hydra.main(config_path="config", config_name="learn_labeler.yaml")
def learn_labeler_and_model(opt):
    opt = parse_option(opt)
    with open_dict(opt):
        opt.old_model_name = opt.model_name
        opt.model_name = f"{opt.model_name}_sup_best_labeler_q{opt.std_factor}"
        opt.test_C = 1 / opt.lam

    logger = util.get_logger(opt.logger_name, file_name=f"{opt.logger_name}_{opt.model_name}")
    logger.info(opt)

    train_datasets, _ = get_datasets(opt, "train", opt.rotate_aug)
    meta_train_dataset = get_meta_dataset(
        opt,
        train_datasets,
        no_replacement=opt.no_replacement,
        db_size=opt.train_db_size,
        fixed_db=opt.fixed_db,
        sample_shape="flat",
    )
    # NOTE: we only want one dataloader so we just pass a singleton list
    meta_trainloader = get_dataloaders([meta_train_dataset], 1, opt.num_workers)[0]

    labeler_opt = change_opt(opt, {"model": opt.label_recovery_model})
    backbone = create_model(labeler_opt.model, dataset=labeler_opt.dataset)
    labeler = get_labeler(backbone, meta_trainloader, logger, labeler_opt)

    label_dataset(meta_trainloader, labeler, labeler_opt, logger)
    # Note that accuracy doesn't really make sense for hierarchical labels
    get_labeler_accuracy(meta_trainloader, labeler, logger)

    # We may want to use a different data augmentation strategy at learning than
    # during labelling. The LabelerDataset allows for a different transformation
    # at train time by setting LabelerDataset(transformation=...)
    if opt.sup_data_aug:
        with open_dict(opt):
            opt.data_aug = opt.sup_data_aug

    # transform will depend on the partition, opt.data_aug and opt.dataset
    sup_transform = get_transform(opt, opt.datasets[0], "train")
    logger.info(f"Supervised transform set to: ")
    logger.info(sup_transform)

    labeler_data = LabelerDataset(
        meta_trainloader,
        labeler,
        labeler_opt,
        transform=sup_transform,
        rotate_aug=labeler_opt.sup_rotate_aug,
    )
    logger.info(f"Shape of LabelerDataset: {labeler_data.labels.shape}")
    logger.info(f"Number of datapoints: {labeler_data._len}")
    logger.info(f"Number of classes: {labeler_data.n_cls}")

    labeler_loader = get_dataloaders([labeler_data], labeler_opt.batch_size, labeler_opt.num_workers)[0]

    val_datasets, _ = get_datasets(opt, partition="val", rotate_aug=False)
    valloaders = get_dataloaders(val_datasets, 256, opt.num_workers, shuffle=False)

    backbone = create_model(opt.train_model, dataset=opt.dataset)
    n_cls = labeler_data.n_cls
    if opt.sup_rotate_aug:
        n_cls *= 4

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
        opt, model, labeler_loader, valloaders, optimizer, scheduler, logger, lambda x: x >= opt.lr_decay_epochs[0]
    )


if __name__ == "__main__":
    learn_labeler_and_model()

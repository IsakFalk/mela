import hydra
import torch
from omegaconf import open_dict
from torch import optim as optim

import util
from dataset.base_dataset import LabelerDataset, MetaDataset
from dataset.mix_dataset import MixDataset
from dataset.data_util import get_datasets_mdl, get_datasets, get_transform
from models.util import create_model
from routines import get_labeler_mdl, get_labeler_accuracy, label_dataset, parse_option
from train_routine import Classifier, full_train, get_dataloaders
from util import change_opt

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

class ClusterLoader:
    def __init__(self, opt, partition):
        self.opt = opt

        # Get dictionary of dataset names and number of classes for each dataset
        ds_dict = get_datasets_mdl(opt, partition, opt.rotate_aug)

        # Create meta-datasets, one for each dataset
        self.dataset_names = list(ds_dict.keys())
        train_datasets = [ds_dict[name]["dataset"] for name in ds_dict.keys()]
        self.meta_datasets = [
            MetaDataset(
                ds, opt, no_replacement=True, db_size=None, fixed_db=None, sample_shape="flat"
            )
            for ds in train_datasets
        ]
        self.meta_trainloaders = get_dataloaders(self.meta_datasets, 1, opt.num_workers)

        def get_loader(self, name):
            return self.meta_trainloaders[self.dataset_names.index(name)]

        def initialize_cycler(self):
            self.trainloader_iterators = [iter(loader) for loader in self.meta_trainloaders]

        def sample_batch(self):
            """Sample batches in a cycled manner from all datasets."""
            trainloader_iterators = [iter(loader) for loader in self.meta_trainloaders]
            while trainloader_iterators:
                for i, iterator in enumerate(trainloader_iterators):
                    try:
                        return next(iterator) # Get batch from dataset i if not exhausted
                    except StopIteration:
                        trainloader_iterators.pop(i)
            return None

@hydra.main(config_path="config", config_name="learn_labeler.yaml")
def learn_labeler_and_model(opt):
    opt = parse_option(opt)
    with open_dict(opt):
        opt.old_model_name = opt.model_name
        opt.model_name = f"{opt.model_name}_sup_best_labeler_q{opt.std_factor}"
        opt.test_C = 1 / opt.lam

    logger = util.get_logger(opt.logger_name, file_name=f"{opt.logger_name}_{opt.model_name}")
    logger.info(opt)

    # Cluster loader allows for more fine-grained sampling for the labeler
    cluster_loader = ClusterLoader(opt, "train", opt.train_batch_sizes)

    labeler_opt = change_opt(opt, {"model": opt.label_recovery_model})
    backbone = create_model(labeler_opt.model, dataset=labeler_opt.dataset)
    labeler = get_labeler_mdl(backbone, cluster_loader, logger, labeler_opt) # TODO: Change get_labeler to fit with new better sampling

    # We may want to use a different data augmentation strategy at learning than
    # during labelling. The LabelerDataset allows for a different transformation
    # at train time by setting LabelerDataset(transformation=...)
    if opt.sup_data_aug:
        with open_dict(opt):
            opt.data_aug = opt.sup_data_aug

    # transform will depend on the partition, opt.data_aug and opt.dataset
    sup_transform = get_transform(opt, opt.datasets[0], "train")
    logger.info("Supervised transform set to: ")
    logger.info(sup_transform)

    labeler_datasets = []
    for name, loader in zip(cluster_loader.meta_trainloaders, cluster_loader.dataset_names):
        labeler_data = LabelerDataset(
            loader,
            labeler,
            labeler_opt,
            transform=sup_transform,
            rotate_aug=labeler_opt.sup_rotate_aug,
        )
        logger.info(f"Dataset: {name}")
        logger.info(f"Shape of LabelerDataset: {labeler_data.labels.shape}")
        logger.info(f"Number of datapoints: {labeler_data._len}")
        logger.info(f"Number of classes: {labeler_data.n_cls}")
        labeler_datasets.append(labeler_data)

    labeler_loaders = get_dataloaders(labeler_datasets, labeler_opt.batch_size, labeler_opt.num_workers)

    val_datasets, _ = get_datasets(opt, partition="val", rotate_aug=False)
    valloaders = get_dataloaders(val_datasets, 256, opt.num_workers, shuffle=False)

    backbone = create_model(opt.train_model, dataset=opt.dataset)
    n_cls = sum([dl.dataset.n_cls for dl in labeler_loaders])
    # if opt.sup_rotate_aug:
    #     n_cls *= 4

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

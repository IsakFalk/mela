import hydra
import torch
import torch.nn.functional as F
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


TRAIN_BATCH_SIZES = {
    "imagenet": 9 * 7 + 2,  # 9, 2 = divmod(128, 14)
    "aircraft": 9,
    "cub": 9,
    "dtd": 9,
    "fungi": 9,
    "omniglot": 9,
    "quickdraw": 9,
    "vgg": 9,
}


class MetaLoader:
    def __init__(self, opt):
        self.opt = opt

        # Get dictionary of dataset names and number of classes for each dataset
        ds_dict = get_datasets_mdl(opt, "train", opt.rotate_aug)

        # Create dataloader for each of these taking into account the batch size factor
        train_datasets = [ds_dict[name]["dataset"] for name in ds_dict.keys()]
        train_batch_sizes = [TRAIN_BATCH_SIZES[name] for name in ds_dict.keys()]
        trainloaders = get_dataloaders(train_datasets, train_batch_sizes, opt.num_workers)
        self.auto_iter_trainloaders = [AutoIterLoader(loader) for loader in trainloaders]
        self._create_label_offset()
        self._create_mask()

    def _create_label_offset(self):
        label_offset = [0]
        for ll in self.auto_iter_trainloaders.values():
            # Assumes that the labels are contiguous and starts at 0 for each dataset
            label_offset.append(len(np.unique(ll.loader.dataset.labels)))
        self.label_offset = np.cumsum(label_offset)[:-1]

    def _create_mask(self):
        db_to_logit_mask = []
        for i, ll in enumerate(self.auto_iter_trainloaders.values()):
            db_to_logit_mask.extend([i] * len(ll.loader.dataset.labels))
        db_to_logit_mask = np.array(db_to_logit_mask)
        _new_array = []
        for i in range(db_to_logit_mask.shape[0]):
            _new_array.append(db_to_logit_mask == i)
        self.db_to_logit_mask = np.array(_new_array)

    def _local_to_global_label(self, label, db_idx):
        return label + self.label_offset[db_idx]

    def _get_mask(self, db_idx):
        db_idx_ohe = F.one_hot(torch.tensor(db_idx), len(self.auto_iter_trainloaders)).numpy()
        return db_idx_ohe @ self.db_to_logit_mask

    def sample(self):
        xs = []
        ys = []
        db_idx = []

        for db_idx, ll in enumerate(self.auto_iter_trainloaders):
            x, y = ll.sample()
            xs.append(x)
            ys.append(self._local_to_global_label(y, db_idx))
            db_idx.append([db_idx] * len(x))

        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        db_idx = np.concatenate(db_idx, axis=0)
        mask = self._get_mask(db_idx)
        return xs, ys, mask


@hydra.main(config_path="config", config_name="sup_baseline.yaml")
def sup_baseline_main(opt):
    opt = parse_option(opt)
    with open_dict(opt):
        opt.model_name = f"{opt.model_name}_sup_base_best"

    logger = util.get_logger(opt.logger_name, file_name=f"{opt.logger_name}_{opt.model_name}")
    logger.info(opt)
    trainloader = MetaLoader(opt)

    # Val datasets stay the same
    val_datasets, _ = get_datasets(opt, "test", False)
    valloaders = get_dataloaders(val_datasets, 256, opt.num_workers, shuffle=False)

    backbone = create_model(opt.model, dataset=opt.dataset)

    model = MDLClassifier(backbone, opt.feat_dim)
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
        trainloader,
        valloaders,
        optimizer,
        scheduler,
        logger,
        lambda x: x >= opt.lr_decay_epochs[0],
    )


if __name__ == "__main__":
    sup_baseline_main()

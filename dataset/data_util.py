import re

import dataset.imagenet as imagenet
import dataset.mix_dataset as mix_dataset
from dataset.base_dataset import MetaDataset
from dataset.imagenet import ImageNet, TieredImageNet, XImageNet
from dataset.mix_dataset import MetaMixDataset, MixDataset, SingleDataset
from dataset.mix_dataset import default_transform as mix_default_transform


def get_transform(opt, name, partition):
    """Get the correct transform of the dataset"""
    # if "imagenet" in name.lower():
    #     data_module = imagenet
    # elif "aircraft" in name.lower() or "cub" in name.lower() or "flower" in name.lower():
    #     data_module = mix_dataset
    # else:
    #     raise NotImplementedError(name)

    data_module = mix_dataset

    if opt.data_aug and partition.lower() == "train":
        transform = data_module.aug_transform
    else:
        transform = data_module.default_transform

    return transform


ximagenet_pattern = re.compile("^x([1-9][0-9]*)ImageNet$")


# meta-dataset
MD_META_SPLITS = {
    "train": ["aircraft", "cub", "dtd", "fungi", "imagenet", "omniglot", "quickdraw", "vgg"],
    "val": ["aircraft", "cub", "dtd", "fungi", "imagenet", "omniglot", "quickdraw", "vgg", "mscoco"],
    "test": ["aircraft", "cub", "dtd", "fungi", "imagenet", "omniglot", "quickdraw", "vgg", "mscoco", "traffic_sign"]
}

def retrieve_dataset_from_name(opt, name, partition, rotate_aug):
    transform = get_transform(opt, name, partition)
    dataset = SingleDataset(name, opt, partition=partition, transform=transform, rotate_aug=rotate_aug)
    n_cls = dataset.n_cls
    #     n_cls = dataset.n_cls
    # if "aircraft" in name.lower() or "cub" in name.lower() or "flower" in name.lower():
    #     # Create new dataset from mixture
    #     dataset = SingleDataset(name, opt, partition=partition, rotate_aug=rotate_aug)
    #     n_cls = dataset.n_cls
    # elif opt.dataset == "miniImageNet":
    #     # n_cls = 64
    #     dataset = ImageNet(opt, partition=partition, transform=transform, rotate_aug=rotate_aug)
    #     n_cls = dataset.n_cls
    # elif opt.dataset == "tieredImageNet":
    #     # n_cls = 351
    #     dataset = TieredImageNet(opt, partition=partition, transform=transform, rotate_aug=rotate_aug)
    #     n_cls = dataset.n_cls
    # elif ximagenet_pattern.match(opt.dataset):
    #     # n_cls = 640
    #     dataset = XImageNet(opt, partition=partition, transform=transform, rotate_aug=rotate_aug, name=opt.dataset)
    #     n_cls = dataset.n_cls
    # else:
    #     raise NotImplementedError(opt.dataset)
    return dataset, n_cls


def get_datasets(opt, partition, rotate_aug):
    """Get the correct (mixed) dataset according to opt, partition and rotate_aug.

    Get the dataset from opt.dataset, partition and rotate aug. If
    the `partition == train` and `opt.data_aug == True` then we
    use use the `aug_transform` from the module instead of the `default_transform`."""
    _datasets = []
    n_cls_counter = []
    datasets = opt.datasets
    # Filter out datasets that are not in the meta split
    if opt.dataset == "metadataset":
        datasets = [dataset for dataset in opt.datasets if dataset in MD_META_SPLITS[partition]]
    for name in datasets:
        ds, n_cls = retrieve_dataset_from_name(opt, name, partition, rotate_aug)
        _datasets.append(ds)
        n_cls_counter.append(n_cls)
    n_cls = sum(n_cls_counter)
    return _datasets, n_cls


def get_meta_dataset(opt, datasets, no_replacement=True, db_size=100, fixed_db=True, sample_shape="few_shot"):
    """Take all BaseDatasets and make them into a mixture dataset

    From n datasets with dataset i having m_i datapoints, we get a dataset
    such that for an index j, we check get the k such that
    sum((m_t)_{t=1}^k) < j \leq sum((m_t)_{t=1}^k+1), i.e.
    datasets[j] = ds_k[j - sum((m_t)_{t=1}^k)]"""
    meta_dataset = MetaMixDataset(
        [
            MetaDataset(
                ds, opt, no_replacement=no_replacement, db_size=db_size, fixed_db=fixed_db, sample_shape=sample_shape
            )
            for ds in datasets
        ]
    )
    return meta_dataset


def get_flat_mixed_dataset(datasets, opt, partition, rotate_aug, transform=mix_default_transform):
    return MixDataset(datasets, opt, partition=partition, transform=transform, rotate_aug=rotate_aug)

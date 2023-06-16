import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from scipy.stats import mode
from tqdm import tqdm

import util
from meta_labeler import MetaLabeler


def listify_parse_option(option, coerce_type=int):
    if isinstance(option, str) and "," in option:
        ll = option.replace(" ", "").split(",")
        return list(map(coerce_type, ll))
    else:
        return coerce_type(option)


def parse_option(opt):
    # Hydra by default makes the config frozen so that
    # we can't add additional key-value pairs to it. This is
    # good since it means we won't fail silently.

    # the open_dict context manager disables this in the scope of
    # the with statement.
    with open_dict(opt):
        if "lr_decay_epochs" in opt:
            iterations = opt.lr_decay_epochs.split(",")
            opt.lr_decay_epochs = list(map(int, iterations))
        if "label_n_ways" in opt:
            opt.label_n_ways = listify_parse_option(opt.label_n_ways)
        if "label_n_shots" in opt:
            opt.label_n_shots = listify_parse_option(opt.label_n_shots)

        if "label_recovery_model" and "train_model" in opt:
            opt.model_name = f"lab:{opt.label_recovery_model}_tr:{opt.train_model}_{opt.dataset}"
        else:
            opt.model_name = f"{opt.model}_{opt.dataset}"
        # if "datasets" in opt:
        #     opt.datasets = listify_parse_option(opt.datasets, str)
        # if "datasets" not in opt or opt.datasets == "":
        #     opt.datasets = [opt.dataset]
        if "dataset" in opt:
            opt.datasets = listify_parse_option(util.map_dataset_to_config_form(opt.dataset), str)
            if not OmegaConf.is_list(opt.datasets):
                opt.datasets = [opt.datasets]
            # Internally used names
            if opt.dataset == "mini60":
                opt.dataset = "x60ImageNet"
            elif opt.dataset == "tiered780":
                opt.dataset = "x780ImageNet"
        if "train_db_size" in opt:
            opt.model_name += f"_f{opt.train_db_size}"
        if "epochs" in opt:
            opt.model_name += f"_e{opt.epochs}"

        if "n_ways" in opt:
            opt.n_ways = listify_parse_option(opt.n_ways)
            opt.model_name += f"_nways{opt.n_ways}"
        if "n_shots" in opt:
            opt.n_shots = listify_parse_option(opt.n_shots)
            opt.model_name += f"_nshots{opt.n_shots}"
        if "data_aug" in opt:
            if opt.data_aug:
                opt.model_name += "_data-aug"
        if "rotate_aug" in opt:
            if opt.rotate_aug:
                opt.model_name += "_rot-aug"
        if "no_replacement" in opt:
            if opt.no_replacement:
                opt.model_name += "_no-replace"
        if "sim_imbalance" in opt:
            if opt.sim_imbalance:
                opt.model_name += "_imba"

        opt.model_name += f"_{opt.trial}"
        opt.model_name = opt.model_name.replace(" ", "")  # Remove whitespace
        opt.model_name = opt.model_name.replace(",", "-")  # Remove whitespace

        if "val_n_shots" in opt:
            opt.val_n_shots = listify_parse_option(opt.val_n_shots)
            if isinstance(opt.val_n_shots, int):
                opt.val_n_shots = [opt.val_n_shots]

        opt.model_path = str(Path(os.getenv("WORKSPACE")) / "checkpoint" / "label_learn" / "saves")
        opt.n_gpu = torch.cuda.device_count()
        # opt.data_root = os.path.expanduser(opt.data_root)

    return opt


def get_labeler(backbone, train_loader, logger, opt):
    """Set up and output the correct labeler using arguments"""
    model = MetaLabeler(backbone, opt, opt.feat_dim).cuda()

    # Load backbone (meta representation) if already trained
    save_dict = torch.load(os.path.join(opt.model_path, opt.pretrained_labeler))["model"]
    util.partial_reload(model, save_dict)

    model.eval()
    # Load centroids if MetaLabeler also already trained
    # else train labeler
    if opt.pretrained_centroids:
        model.load_cluster(f"{opt.model_path}/{opt.pretrained_centroids}")
        logger.info(f"Loaded pretrained centroids successfully, number of clusters: {model.K}")
    else:
        logger.info("Training labeler")
        tmp = []
        # init centroids
        for id, batch_data in enumerate(train_loader):
            if id >= model.K / model.n_ways:
                break
            task_data = list(map(lambda x: x[0], batch_data))
            xs, _, real_cls, _ = task_data
            tmp.append(real_cls)

            model.init_centroid(*util.to_cuda_list([xs, real_cls]))

        logger.info(f"init done {np.unique(np.concatenate(tmp)).shape}")

        prev_k = 0
        while model.K != prev_k:
            prev_k = model.K
            clustered = 0
            for id, batch_data in enumerate(tqdm(train_loader)):
                task_data = list(map(lambda x: x[0], batch_data))
                xs, _, real_cls, _ = task_data
                success = model.cluster_task(*util.to_cuda_list([xs, real_cls]))
                clustered += int(success)
            logger.info(f"Clustering rate: {clustered / len(train_loader.dataset)}")
            model.remove_cluster(opt.n_ways, opt.std_factor)
            logger.info(f"No. of clusters: {model.K}")

        torch.save(
            model.centroid,
            f"{opt.model_path}/{opt.old_model_name}_centroids_c{model.K}_q{opt.std_factor}",
        )
    return model

def get_labeler_mdl(backbone, cluster_train_loader, logger, opt):
    """Set up and output the correct labeler using arguments"""
    model = MetaLabeler(backbone, opt, opt.feat_dim).cuda()

    # Load backbone (meta representation) if already trained
    save_dict = torch.load(os.path.join(opt.model_path, opt.pretrained_labeler))["model"]
    util.partial_reload(model, save_dict)

    model.eval()
    # Load centroids if MetaLabeler also already trained
    # else train labeler
    if opt.pretrained_centroids:
        model.load_cluster(f"{opt.model_path}/{opt.pretrained_centroids}")
        logger.info(f"Loaded pretrained centroids successfully, number of clusters: {model.K}")
    else:
        # First just try to do this with ilsvrc
        logger.info("Training labeler")
        # init centroids
        for id, batch_data in enumerate(cluster_train_loader.get_loader("imagenet")):
            if id >= model.K / model.n_ways:
                break
            task_data = list(map(lambda x: x[0], batch_data))
            xs, _, real_cls, _ = task_data
            # Initialize K centroids using the tasks
            model.init_centroid(*util.to_cuda_list([xs, real_cls]))

        # Train
        def stop_criteria(prev_k, curr_k):
            return prev_k == curr_k

        prev_k = 0
        while not stop_criteria(prev_k, model.K):
            batch_data = cluster_train_loader.sample_batch()
            prev_k = model.K
            clustered = 0
            count = 0
            while batch_data is not None:
                task_data = list(map(lambda x: x[0], batch_data))
                xs, _, real_cls, _ = task_data
                success = model.cluster_task(*util.to_cuda_list([xs, real_cls]))
                count += 1
                clustered += int(success)
                batch_data = cluster_train_loader.sample_batch()
            logger.info(f"Clustering rate: {clustered / count}")
            model.remove_cluster(opt.n_ways, opt.std_factor)
            logger.info(f"No. of clusters: {model.K}")

        torch.save(
            model.centroid,
            f"{opt.model_path}/{opt.old_model_name}_centroids_c{model.K}_q{opt.std_factor}",
        )
    return model


def get_labeler_accuracy(dataloader, labeler, logger):
    """Get accuracy of labeler"""

    # Since we use the dataloader we need to build the dataset again
    labels = []
    pseudo_labels = []

    # Iterate through all of the batches and create the mapping
    # from label to pseudo_label. The index of the list acts as
    # a mapping from x_i to y_i and pseudo_y_i
    for batch_data in tqdm(dataloader):
        # extract data and get the true classes of the instances
        task_data = list(map(lambda x: x[0], batch_data))
        xs, _, real_cls, _ = task_data
        real_cls = real_cls.numpy().tolist()

        # Add true labels
        labels.extend(real_cls)

        # label xs and add cluster labels
        _, pseudo_cls = labeler.label_samples(xs.cuda())
        if pseudo_cls is not None:
            pseudo_cls = pseudo_cls.cpu().numpy().tolist()
            # Sanity check
            assert len(pseudo_cls) == len(real_cls)
            pseudo_labels.extend(pseudo_cls)
        else:
            # if pseudo_cls is None we just use that as the
            # pseudo label and handle it later
            pseudo_labels.extend([None] * len(real_cls))
    labels = np.array(labels)
    pseudo_labels = np.array(pseudo_labels)
    # To get accuracy we create the mapping from x -> f(x)
    # where f is a classifier induced by te clustering algorithm
    # that clusters x according to the majority label in the cluster of x
    # We now create the dictionary which maps from pseudo_labs to the most common class of that centroid
    mapping = {}

    # Remove None since np.unique errors out
    remove_none_idx = pseudo_labels != None
    no_none_pseudo_labels = pseudo_labels[remove_none_idx].astype(int)
    for pslab in np.unique(no_none_pseudo_labels):
        # extract the index of pslab
        idx = no_none_pseudo_labels == pslab
        x = labels[remove_none_idx][idx]
        # None is not an okay ground truth label
        labels_for_pslab = x[x != None].astype(int)

        # Get the most frequent label for cluster
        mapping[pslab] = mode(labels_for_pslab)[0][0]

    # Add None as -1 since it's never correct
    mapping[None] = -1

    pred_labels = np.vectorize(mapping.get)(pseudo_labels)
    acc = (pred_labels == labels).mean()

    logger.info(f"Clustering accuracy: {acc:.3f}")


def label_dataset(meta_trainloader, labeler, opt, logger):
    """Output information about how well the clustering step is doing"""

    clusters = defaultdict(list)
    sub_classes = defaultdict(list)

    for batch_data in tqdm(meta_trainloader):
        task_data = list(map(lambda x: x[0], batch_data))
        xs, _, real_cls, _ = task_data
        real_cls = real_cls.numpy().tolist()

        _, pseudo_cls = labeler.label_samples(xs.cuda())
        if pseudo_cls is not None:
            pseudo_cls = pseudo_cls.cpu().numpy().tolist()

            assert len(pseudo_cls) == len(real_cls)
            for pseudo_y, y in zip(pseudo_cls, real_cls):
                clusters[pseudo_y].append(y)
                sub_classes[y].append(pseudo_y)

    logger.info(f"number of real classes {len(sub_classes)}")
    for key, val in sub_classes.items():
        val = set(val)
        if len(val) > 1:
            tmp = []
            for cls in val:
                tmp.append(len(clusters[cls]))
            logger.info(f"class {key} is split between {val}, with count {tmp}")

    for key in clusters.keys():
        clss, counts = np.unique(clusters[key], return_counts=True)
        if len(clss) > 1:
            logger.info(f"cluster {key} contains classes {clss} with distribution of {counts/len(clusters[key])} ")

import random
import socket
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import eval_util
import util
from eval_util import mean_confidence_interval


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


class Classifier(torch.nn.Module):
    """Used for training the backbone during meta-train using CE

    Wrapper to train the backbone during meta-train. At meta-test
    time the backbone is extracted and we use a different strategy
    given by the base_algorithm defined in this class."""

    def __init__(self, backbone, feat_dim, n_cls):
        super().__init__()
        self.backbone = backbone
        self.layer = nn.Linear(feat_dim, n_cls)

    def forward(self, xs, ys):
        feats = self.backbone(xs)
        logits = self.layer(feats)
        loss = F.cross_entropy(logits, ys)

        return loss


class SemanticMultiheadClassifier(torch.nn.Module):
    """Used for training the backbone during meta-train using semantic CE

    Wrapper to train the backbone during meta-train. At meta-test
    time the backbone is extracted and we use a different strategy
    given by the base_algorithm defined in this class."""

    def __init__(self, backbone, feat_dim, n_cls):
        super().__init__()
        self.backbone = backbone
        self.n_cls = n_cls
        summed_n_cls = sum(n_cls)
        self.layer = nn.Linear(feat_dim, summed_n_cls)

    def _prepare_logits_and_ys(self, logits, ys):
        logits1 = logits[:, : self.n_cls[0]]
        logits2 = logits[:, self.n_cls[0] : self.n_cls[0] + self.n_cls[1]]
        logits3 = logits[:, self.n_cls[0] + self.n_cls[1] :]
        ys1, ys2, ys3 = ys[:, 0], ys[:, 1], ys[:, 2]
        return logits1, logits2, logits3, ys1, ys2, ys3

    def forward(self, xs, ys):
        ys1, ys2, ys3 = ys[:, 0], ys[:, 1], ys[:, 2]
        feats = self.backbone(xs)
        logits = self.layer(feats)
        logits1, logits2, logits3, ys1, ys2, ys3 = self._prepare_logits_and_ys(logits, ys)
        loss = 0.0
        loss += F.cross_entropy(logits1, ys1)
        loss += F.cross_entropy(logits2, ys2)
        loss += F.cross_entropy(logits3, ys3)
        return loss / 3.0


def extract_batch_data_to_cuda(batch_data):
    # Depending on if we are in few-shot or flat / SL setting
    if len(batch_data) == 5:
        batch_data = util.to_cuda_list(batch_data[:4])
    elif len(batch_data) == 3:
        batch_data = util.to_cuda_list(batch_data[:2])
    elif len(batch_data) == 2:
        # In this case we are pushing two tasks (one sl, one fs)
        batch_data = extract_batch_data_to_cuda(batch_data[0]), extract_batch_data_to_cuda(batch_data[1])
    return batch_data


def np_check_nan_or_inf(x):
    return np.isnan(x).any() or np.isinf(x).any()


def torch_check_nan_or_inf(x):
    return torch.isnan(x).any() or torch.isinf(x).any()


def train(model, train_loader, optimizer, logger, opt=None, progress=False):
    avg_metric = util.AverageMeter()

    if progress:
        train_loader = tqdm(train_loader)

    for i, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        batch_data = extract_batch_data_to_cuda(batch_data)
        loss = model.forward(*batch_data)
        # Guard against nan / inf loss
        nan_loss = torch_check_nan_or_inf(loss)
        counter = 0
        if nan_loss:
            logger.info(f"Corrupt loss found in batch {i}")
        while nan_loss and counter <= 5:
            optimizer.zero_grad()
            counter += 1
            loss = model.forward(*batch_data)
            nan_loss = torch_check_nan_or_inf(loss)
        if counter > 5:
            logger.info(f"Loss still corrupted after retrying {counter} times, skipping batch")
            continue
        elif counter > 0:
            logger.info(f"Retrying batch succeeded on try {counter}!")
        loss.backward()
        optimizer.step()
        avg_metric.update([loss.item()])

    return avg_metric.avg


def test_fn(model, test_loaders, val_n_shots, logger, opt=None):
    accs = []

    model.eval()
    names = []
    for i, loader in enumerate(test_loaders):
        acc = eval_util.meta_test_new(model, loader, val_n_shots, logger, opt=opt, feats=None)
        accs.append(acc)
        names.append(loader.dataset.name)
    model.train()
    # Per dataset acc
    per_dataset_accs = {name: mean_confidence_interval(np.array(acc)) for (name, acc) in zip(names, accs)}
    # Average
    accs = np.concatenate(accs).flatten()
    total_acc = mean_confidence_interval(accs)
    return total_acc, per_dataset_accs, None


def full_train(
    opt,
    model,
    train_loader,
    test_loaders,
    optimizer,
    lr_sch,
    logger,
    eval_cond,
    test_fn=test_fn,
):
    best_val_acc = {key: 0 for key in opt.val_n_shots}
    for epoch in range(opt.epochs):
        train_loss = train(model, train_loader, optimizer, logger, opt=opt, progress=opt.progress)
        if lr_sch:
            lr_sch.step()

        logger.info(f"epoch {epoch}")
        info = util.print_metrics(["Loss"], train_loss)
        logger.info(info)

        # setting feats_list to None means that test_fn will return a list
        # with the correct features for each loader
        # feats_list = None
        if eval_cond(epoch):
            for curr_val_n_shots, best_acc in best_val_acc.items():
                test_acc = test_fn(model, test_loaders, curr_val_n_shots, logger, opt=opt)[0][0]
                logger.info(f"{curr_val_n_shots}nshots val acc: {test_acc}")
                if test_acc > best_acc:
                    logger.info(f"Best acc so far, saving model...")
                    best_val_acc[curr_val_n_shots] = test_acc
                    util.save_routine(
                        epoch, model, optimizer, f"{opt.model_path}/{opt.model_name}_xvalnshots{curr_val_n_shots}_best"
                    )

    for curr_val_n_shots, best_acc in best_val_acc.items():
        logger.info(f"Best acc for {curr_val_n_shots}-shots xval: {best_acc}")
        logger.info(f"Saved model: {opt.model_name}_xvalnshots{curr_val_n_shots}_best")

    # Output final information
    logger.info(f"Host: {socket.gethostname()}")
    logger.info(f"Git-hash: {get_git_revision_hash()}")


def init_worker_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(datasets, batch_size, n_workers, shuffle=True, drop_last=False):
    dataloaders = []
    for ds in datasets:
        dataloaders.append(
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=n_workers,
                pin_memory=True,
                worker_init_fn=init_worker_fn,
            )
        )
    return dataloaders

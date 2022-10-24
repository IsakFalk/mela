import os

import hydra
import torch
from omegaconf import open_dict
from torch import nn
from torch import optim as optim

import eval_util
import util
from dataset.data_util import get_datasets, get_meta_dataset
from meta_learner import MetaLS
from models.resfc import ResFC
from models.util import create_model
from routines import parse_option
from train_routine import full_train, get_dataloaders


class moduleWrapper(nn.Module):
    def __init__(self, backbone, trans):
        super().__init__()

        self.backbone = backbone
        self.trans = trans

    def forward(self, xs):
        feat = self.backbone(xs)
        feat = eval_util.normalize(feat)
        return self.trans(feat)

    def train(self, mode=True):
        self.trans.train(mode)
        return self


@hydra.main(config_path="config", config_name="fine_tune.yaml")
def fine_tune_main(opt):
    util.set_up_cudnn()
    opt = parse_option(opt)
    with open_dict(opt):
        opt.model_name = f"{opt.model_name}_fine_tune"

    logger = util.get_logger(opt.logger_name, file_name=f"{opt.logger_name}_{opt.model_name}")
    logger.info(opt)

    train_datasets, n_cls = get_datasets(opt, "train", opt.rotate_aug)
    meta_train_dataset = get_meta_dataset(opt, train_datasets)
    # NOTE: we only want one dataloader so we just pass a singleton list
    meta_trainloader = get_dataloaders([meta_train_dataset], 1, opt.num_workers)[0]

    val_datasets, _ = get_datasets(opt, "val", False)
    valloaders = get_dataloaders(val_datasets, 256, opt.num_workers, shuffle=False)

    save_dicts = torch.load(os.path.join(opt.model_path, opt.pretrained_model))
    model_params = save_dicts["model"]
    model_params = util.change_param_prefix(model_params, "module", "backbone")

    backbone = create_model(opt.model, dataset=opt.dataset)
    backbone.eval()
    ft_model = ResFC(opt.feat_dim, opt.feat_dim, residual=True, layer_norm=True)

    # loading the backbone here, avoiding changing saved parameter names since composite_backbone adds prefix "backbone"
    composite_backbone = moduleWrapper(backbone, ft_model)
    util.partial_reload(composite_backbone, model_params)

    model = MetaLS(composite_backbone, opt, opt.feat_dim, opt.extra_reg)
    model = model.cuda()

    if opt.rotate_aug:
        n_cls *= 4

    optimizer = optim.AdamW(ft_model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    scheduler = None

    full_train(
        opt,
        model,
        meta_trainloader,
        valloaders,
        optimizer,
        scheduler,
        logger,
        lambda x: x >= 0,
    )


if __name__ == "__main__":
    fine_tune_main()

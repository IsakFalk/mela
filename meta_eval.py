import os

import hydra
import torch
from omegaconf import open_dict

import train_routine
import util
from dataset.data_util import get_datasets
from fine_tune import moduleWrapper
from models.resfc import ResFC
from models.util import create_model
from routines import parse_option
from train_routine import Classifier, get_dataloaders


@hydra.main(config_path="config", config_name="meta_eval.yaml")
def eval_main(opt):
    util.set_up_cudnn()
    opt = parse_option(opt)

    logger = util.get_logger(opt.logger_name, file_name=f"{opt.logger_name}_{opt.model_name}")
    logger.info(opt)

    with open_dict(opt):
        opt.data_aug = False

    test_datasets, _ = get_datasets(opt, partition="test", rotate_aug=False)
    testloaders = get_dataloaders(test_datasets, 256, opt.num_workers, shuffle=False)

    backbone = create_model(opt.model, dataset=opt.dataset)
    backbone.eval()

    if "fine_tune" in opt.pretrained_model:
        ft_model = ResFC(opt.feat_dim, opt.feat_dim, residual=True, layer_norm=True)
        backbone = moduleWrapper(backbone, ft_model)

    # classifier is just an wrapper, not used during testing
    model = Classifier(backbone, opt.feat_dim, 1)

    save_dicts = torch.load(os.path.join(opt.model_path, opt.pretrained_model))
    model_params = save_dicts["model"]
    model_params = util.change_param_prefix(model_params, "module", "backbone")
    util.partial_reload(model, model_params)
    model = model.cuda()
    model.eval()

    n_shots = 1
    with open_dict(opt):
        opt.val_n_shots = n_shots
    total_acc, per_dataset_accs, _ = train_routine.test_fn(model, testloaders, opt.val_n_shots, logger, opt)
    logger.info(f"Average: 1-shot Acc: {total_acc[0]}, Std: {total_acc[1]}")
    if len(testloaders) > 1:
        for name, acc in per_dataset_accs.items():
            logger.info(f"{name}: 1-shot Acc: {acc[0]}, Std: {acc[1]}")

    n_shots = 5
    with open_dict(opt):
        opt.val_n_shots = n_shots
    total_acc, per_dataset_accs, _ = train_routine.test_fn(model, testloaders, opt.val_n_shots, logger, opt)
    logger.info(f"Average: 5-shot Acc: {total_acc[0]}, Std: {total_acc[1]}")
    if len(testloaders) > 1:
        for name, acc in per_dataset_accs.items():
            logger.info(f"{name}: 1-shot Acc: {acc[0]}, Std: {acc[1]}")


if __name__ == "__main__":
    eval_main()

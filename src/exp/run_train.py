
import yaml
import torch
import os
from argparse import Namespace
from codes.supports.utils import *
from codes.train_model import ModelTrainer
import time

torch.backends.cudnn.deterministic = True

config_file = '../config.yaml'
config = path_to_dict(config_file)['exp']

def run_train(
    args: Namespace,
    params,
    seed,
    logger,
):
    """
    Args:
        args (Namespace):
        params (Dict) : Settings.
        seed (int) : Fold number.
        logger (Logger) : Log file.
    Returns:
        acc (float): accuracy.
    """
    settings = args.settings

    save_dir = f'{params["ref_name"]}'+  config['outputs_dir'] + f'/seed{seed}'
    makedirs(save_dir)

    trainer = ModelTrainer(args, params, save_dir, seed, logger)
    acc = trainer.run()
    del trainer
    return acc

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        '--settings', 
        default="./resources/settings.yaml",
        dest='settings')
    
    parser.add_argument(
        '--ref_name',
        default="Pan-Asian_REF",
        dest='ref_name')

    parser.add_argument(
        '--data_dir',
        default="Pan-Asian",
        dest='data_dir')

    parser.add_argument('--device', default="cuda", dest='device')
    args = parser.parse_args()

    settings = path_to_dict(args.settings)
    settings['ref_name'] = args.ref_name
    settings['data_dir'] = args.data_dir
    result_dir = f'{settings["ref_name"]}' + config['outputs_dir']
    makedirs(result_dir)
    log_file = result_dir + '/training.log'
    logger = Logger(log_file)

    for seed in range(settings['fold_num']):
        _ = run_train(args, settings, seed, logger)

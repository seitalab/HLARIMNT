
import yaml
import torch
import os
from argparse import Namespace
from codes.supports.utils import *
from codes.train_model import ModelTrainer
import time

torch.backends.cudnn.deterministic = True

config_file = '../config.yaml'
config = path_to_dict(config_file)

def run_train(
    args: Namespace,
    params,
    seed,
    logger,
    mode
) -> str:
    """
    Args:
        args (Namespace):
        hps_name (str): Name of hyperparameter search target.
            must have keys listed in `MUST_KEYS`. 
        info_string (str):
    Returns:
        save_loc (str):
    """
    settings = args.settings

    save_loc = f'/{params["exp_name"]}'+  f'/{params["task"]}-{params["data"]["dataset"]}' + f'/seed{seed}'
    save_dir = config[mode]['save_root'] + save_loc

    trainer = ModelTrainer(args, params, save_dir, seed, logger, mode)
    acc = trainer.run()
    del trainer
    return acc
    # Prepare save loc.

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--settings', help='path to config file', 
        # default="./resources/trials/trial.yaml",
        dest='settings')

    parser.add_argument('--device', default="cuda", dest='device')
    args = parser.parse_args()

    settings = path_to_dict(args.settings)['base']
    result_dir = config['exp']['save_root'] + f'/{settings["exp_name"]}'+ f'/{settings["task"]}-{settings["data"]["dataset"]}'
    makedirs(result_dir)
    log_file = result_dir + '/training.log'
    logger = Logger(log_file)

    if settings['fold_num'] == -1:
        start_time = time.time()
        logger.log(f'process starts at {start_time}')
        _ = run_train(args, settings, 0, logger, 'exp')
        end_time = time.time()
        logger.log(f'process ends at {end_time}')
        logger.log(f'all processing time is {end_time - start_time} seconds.')
    else:
        for seed in range(settings['fold_num']):
            _ = run_train(args, settings, seed, logger, 'exp')

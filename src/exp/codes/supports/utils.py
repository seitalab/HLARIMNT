
import os
from datetime import datetime
from typing import Dict
from argparse import Namespace
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F

def makedirs(dir_path) -> None:
    """
    Args:
        dir_path:
    Returns:
        None
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def path_to_dict(path: str) -> Dict:
    """
    Arg:
        path (str) : Path to file.
    Returns:
        info_dict (Dict) :
    """
    yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        lambda loader, node: OrderedDict(loader.construct_pairs(node)))
    with open(path) as f:
        info_dict = yaml.safe_load(f)
    return info_dict

class Logger(object):
    def __init__(self, fn):
        self.f = open(fn, 'w')

    def log(self, msg, *args, **kwargs):
        msg = msg.format(*args, **kwargs)
        print(msg)
        self.f.write(msg+"\n")

def nll(pred, gt, val=False):
    if val:
        return F.nll_loss(pred, gt, size_average=False)
    else:
        return F.nll_loss(pred, gt)

def get_loss(num_task):
    loss_fn = {}
    for t in range(num_task):
        loss_fn[t] = nll
    return loss_fn


import torch
import torch.nn as nn
from argparse import Namespace
from torch import Tensor
from typing import Tuple, Dict
import yaml
from importlib import import_module

modeldict_file = './resources/models.yaml'
with open(modeldict_file) as f:
    modeldict = yaml.safe_load(f)

cfg_file = '../config.yaml'
with open(cfg_file) as f:
    config = yaml.safe_load(f)

def _get_num_classes(dataset: str, hla: str, digit: str) -> int:
    '''
    Args:
        dataset (str) : Name of dataset
        hla (str) : Name of HLA gene
        digit (str) : Name of digit
    '''
    assert digit in ['2-digit', '4-digit', '6-digit']

    data_root = config['dataset']['save_root']
    data_dir = config['dataset'][dataset]['dirname']
    hla_info = config['dataset'][dataset]['hla_info']
    with open(data_root + data_dir + hla_info) as f:
        hladict = yaml.safe_load(f)

    allele_list = hladict[hla][digit]

    num_classes = len(allele_list)
    return num_classes

class PredictorShared(nn.Module):
    def __init__(self, 
                foot: nn.Module,
                pe: nn.Module,
                shared: nn.Module,
    ) -> None:
        super(PredictorShared, self).__init__()

        self.foot = foot
        self.shared = shared
        self.pe = pe

    def forward(self, x, filters) -> Tuple[Tensor, Dict]:
        '''
        Args:
            x (Tensor): Tensor of size (batchsize, num_ref, 2)
            params (Dict): dict of filters for CNN model 
        Returns:
            x (Tensor) : Tensor of size (batchsize, num_classes)
            params (Dict): dict of filters for CNN model
        '''
        x, filters = self.foot(x, filters)
        x, filters = self.pe(x, filters)
        x, filters = self.shared(x, filters)
        return x, filters

class PredictorEach(nn.Module):
    def __init__(self, 
                each: nn.Module
    ) -> None:
        super(PredictorEach, self).__init__()
        self.each = each

    def forward(self, x, filters) -> Tuple[Tensor, Dict]:
        '''
        Args:
            x (Tensor): Tensor of size (batchsize, num_ref, 2)
            params (Dict): dict of filters for CNN model 
        Returns:
            x (Tensor) : Tensor of size (batchsize, num_classes)
            params (Dict): dict of filters for CNN model
        '''
        x, filters = self.each(x, filters)
        return x, filters

def implement_model(dataset_name, hla_list, digit, input_len, chunk_len, settings):
    models = {}
    models['shared'] = _implement_model_shared(settings, input_len, chunk_len)
    for hla in hla_list:
        class_num = _get_num_classes(dataset_name, hla, digit)
        models[hla] = _implement_model_each(settings, class_num)
    return models

def _implement_model_shared(settings: Dict, input_len, chunk_len) -> nn.Module:
    foot_type = settings['model']['foot']['type']
    module_name_f = modeldict['foot'][foot_type]['module_file']
    class_name_f = modeldict['foot'][foot_type]['class_name']
    foot_module = import_module(f'codes.architectures.foots.{module_name_f}')
    Foot = getattr(foot_module, class_name_f)
    foot = Foot(settings, input_len, chunk_len)

    pe_type = settings['model']['pos_encode']['type']
    module_name_p = modeldict['pos_encode'][pe_type]['module_file']
    class_name_p = modeldict['pos_encode'][pe_type]['class_name']
    pe_module = import_module(f'codes.architectures.pos_encode.{module_name_p}')
    PE = getattr(pe_module, class_name_p)
    pe = PE(settings, input_len)    

    shared_type = settings['model']['shared']['type']
    module_name_s = modeldict['shared'][shared_type]['module_file']
    class_name_s = modeldict['shared'][shared_type]['class_name']
    shared_module = import_module(f'codes.architectures.shared_net.{module_name_s}')
    Shared = getattr(shared_module, class_name_s)
    
    shared = Shared(settings, input_len)

    predictor = PredictorShared(foot, pe, shared)

    return predictor

def _implement_model_each(settings: Dict, class_num: int) -> nn.Module:
    each_type = settings['model']['each']['type']
    module_name_e = modeldict['each'][each_type]['module_file']
    class_name_e = modeldict['each'][each_type]['class_name']
    each_module = import_module(f'codes.architectures.each_net.{module_name_e}')
    Each = getattr(each_module, class_name_e)
    each = Each(settings, class_num)

    predictor = PredictorEach(each)

    return predictor
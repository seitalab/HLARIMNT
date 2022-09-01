
import torch
import torch.nn as nn
from torch import Tensor
import yaml
import json
from importlib import import_module
from typing import List, Dict

modeldict_file = './resources/models.yaml'

with open(modeldict_file) as f:
    modeldict = yaml.safe_load(f)

cfg_file = '../config.yaml'
with open(cfg_file) as f:
    config = yaml.safe_load(f)['exp']

def _get_num_classes(dataset: str, hla: str, digit: str) -> int:
    """
    Args:
        dataset (str) : Name of dataset
        hla (str) : Name of HLA gene
        digit (str) : Name of digit
    Returns:
        num_classes (int) : Numbers of the classes of the layer.
    """
    assert digit in ['2-digit', '4-digit']

    hla_info_loc = f'codes/data/{dataset}/hla_info.json'
    with open(hla_info_loc) as f:
        hladict = json.load(f)

    allele_list = hladict[hla][digit]
    num_classes = len(allele_list)
    return num_classes

class PredictorShared(nn.Module):
    def __init__(self, 
                embedding: nn.Module,
                transformer: nn.Module,
    ) -> None:
        super(PredictorShared, self).__init__()

        self.embedding = embedding
        self.transformer = transformer

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of size (batch_size, chunk_num+1, chunk_len, 2)
        Returns:
            x (Tensor) : Tensor of size (batch_size, emb_dim,)
        """
        x = self.embedding(x)
        x = self.transformer(x)
        return x

class PredictorEach(nn.Module):
    def __init__(self, 
                classification: nn.Module
    ) -> None:
        super(PredictorEach, self).__init__()
        self.classification = classification

    def forward(self, x) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of size (batch_size, emb_dim,)
        Returns:
            x (Tensor) : Tensor of size (batchsize, class_num)
        """
        x = self.classification(x)
        return x

def implement_model(ref_name: str, hla_list: List, digit: str, input_len: int, chunk_len: int, settings: Dict):
    """
    Args:
        ref_name (str) : Name of dataset used for training.
        hla_list (List) : List of HLA genes the genotypes of which the model predicts.
        digit (str) : digit of HLA.
        input_len (int) : Length of the input (chunk_num+1).
        chunk_len (int) : Numbers of SNPs one chunk has.
        settings (Dict) : Settings of the experiment.
    Returns:
        models (Dict) : whole model.
    """
    models = {}
    models['shared'] = _implement_model_shared(settings, input_len, chunk_len)
    for hla in hla_list:
        class_num = _get_num_classes(ref_name, hla, digit)
        models[hla] = _implement_model_each(settings, class_num)
    return models

def _implement_model_shared(settings: Dict, input_len: int, chunk_len: int) -> nn.Module:
    """
    Args:
        settings (Dict) : Settings of the experiment.
        input_len (int) : Length of the input (chunk_num+1).
        chunk_len (int) : Numbers of SNPs one chunk has.
    Returns:
        predictor (nn.Module) : Shared part of the model.
    """
    emb_type = settings['model']['embedding']['type']
    module_name_e = modeldict['EmbeddingLayer'][emb_type]['module_file']
    class_name_e = modeldict['EmbeddingLayer'][emb_type]['class_name']
    emb_module = import_module(f'codes.architectures.EmbeddingLayer.{module_name_e}')
    Embedding = getattr(emb_module, class_name_e)
    embedding = Embedding(settings, input_len, chunk_len)

    transformer_type = settings['model']['transformer']['type']
    module_name_t = modeldict['TransformerLayer'][transformer_type]['module_file']
    class_name_t = modeldict['TransformerLayer'][transformer_type]['class_name']
    transformer_module = import_module(f'codes.architectures.TransformerLayer.{module_name_t}')
    Transformer = getattr(transformer_module, class_name_t)
    transformer = Transformer(settings, input_len)

    predictor = PredictorShared(embedding, transformer)
    return predictor

def _implement_model_each(settings: Dict, class_num: int) -> nn.Module:
    classification_type = settings['model']['classification']['type']
    module_name_c = modeldict['ClassificationLayer'][classification_type]['module_file']
    class_name_c = modeldict['ClassificationLayer'][classification_type]['class_name']
    classification_module = import_module(f'codes.architectures.ClassificationLayer.{module_name_c}')
    Classification = getattr(classification_module, class_name_c)
    classification = Classification(settings, class_num)

    predictor = PredictorEach(classification)
    return predictor

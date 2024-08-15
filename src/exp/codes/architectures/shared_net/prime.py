
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from argparse import Namespace
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor

class PrimeShared(nn.Module):
    def __init__(
        self, params, input_len) -> None:
        """
        Args:
            params (dict): dictionary of hyper parameter
        Returns:
            None
        """
        super(PrimeShared, self).__init__()

        emb_dim = params['emb_dim']
        dim_feedforward = params['dim_feedforward']
        nhead = params['nhead']
        dropout = params['dropout']
        num_layers = params['num_layers']
        self.input_collapsed = params['input_collapse']
        fc_len = params['emb_dim']
        self.params = params

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, dim_feedforward=dim_feedforward, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
        #print(self.encoder)
        self.fc = nn.Linear(emb_dim, fc_len) ###concatするなら*3する
        key_list = [f'layers.{id}' for id in range(num_layers)]
        value_list = [f'feature{id}' for id in range(num_layers)]
        self.features =  dict(zip(key_list, value_list))

        #self.feature_extractor = create_feature_extractor(self.encoder, self.features)

    def forward(self, x: Tensor, filters) -> Tensor:
        """
        Args:
            x (Tensor):
        Returns:
            x (Tensor):
        """
        if self.training and self.input_collapsed:
            mask_input = torch.bernoulli(x.data.new(x.data.size()).fill_(random.uniform(0.8, 1)))
            x = x * mask_input

        x = self.encoder(x)
        #x_dict = self.feature_extractor(x)
        #x = torch.cat((x_dict['feature0'][:,0,:], x_dict['feature1'][:,0,:], x_dict['feature2'][:,0,:]), 1)
        #x = (x_dict['feature0'][:,0,:] + x_dict['feature1'][:,0,:] + x_dict['feature2'][:,0,:]) / 3.0
        if self.params['encode'] == 'input_conv':
            x = x.view(x.shape[0], -1)
        else:
            x = x[:,0,:]
        #x = self.fc(x)
        return x, filters

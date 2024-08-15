
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from torch import Tensor
from argparse import Namespace
from typing import Tuple, Dict

class BiRNNShared(nn.Module):
    def __init__(self, params, input_len):
        super(BiRNNShared, self).__init__()

        self.layer_num = params['layer_num_rnn']
        emb_dim = params['emb_dim']
        hidden_dim = params['emb_dim']

        self.first_layer_fw = FwRNNLayer(params)
        self.first_layer_bw = BwRNNLayer(params)
        fw_layers = []
        bw_layers = []

        for layer_id in range(self.layer_num):
            fw_rnn = FwRNNLayer(params)
            bw_rnn = BwRNNLayer(params)
            fw_layers.append(fw_rnn)
            bw_layers.append(bw_rnn)
        self.fw_layers = nn.ModuleList(fw_layers)
        self.bw_layers = nn.ModuleList(bw_layers)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=self.layer_num, bidirectional=True, batch_first=True)

    def forward(self, x, filters):
        '''
        fw_out, fw_hc = self.first_layer_fw(x, filters)
        bw_out, bw_hc = self.first_layer_bw(x, filters)

        for layer_id in range(self.layer_num-1):
            l_fw = self.fw_layers[layer_id]
            l_bw = self.bw_layers[layer_id]
            fw_out, fw_hc = l_fw(fw_out, filters)
            bw_out, bw_hc = l_bw(bw_out, filters)

        x = torch.cat([fw_hc[0], bw_hc[0]], dim=1)
        '''
        out, hc = self.gru(x)
        x = torch.cat([hc[0], hc[1]], dim=1)

        return x, filters

class FwRNNLayer(nn.Module):
    def __init__(self, params):
        super(FwRNNLayer, self).__init__()

        emb_dim = params['emb_dim']
        hidden_dim = params['emb_dim']

        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, x, filters):
        fw_out, fw_hc = self.gru(x)
        fw_out += x
        return fw_out, fw_hc

class BwRNNLayer(nn.Module):
    def __init__(self, params):
        super(BwRNNLayer, self).__init__()

        emb_dim = params['emb_dim']
        hidden_dim = params['emb_dim']

        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, x, filters):
        for i in range(x.shape[0]):
            tmp = x[i].flipud()
            x[i] = tmp
        bw_out, bw_hc = self.gru(x)
        bw_out += x
        for i in range(bw_out.shape[0]):
            tmp = bw_out[i].flipud()
            bw_out[i] = tmp        
        return bw_out, bw_hc

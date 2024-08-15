
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from argparse import Namespace
from torch import Tensor
from typing import Tuple, Dict

class PrimeEachNet(nn.Module):
    def __init__(self, params, num_class: int) -> None:
        '''
        Args:
            params (dict):
        Returns:
            None
        '''
        super(PrimeEachNet, self).__init__()
        fc_len = params['emb_dim']
        dropout = params['dropout']
        if params['encode'] == 'input_conv':
            self.fc = nn.Linear(params['emb_dim'] * params['chunk_split_num'], num_class)
        else:
            self.fc = nn.Linear(fc_len, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, filters) -> Tuple[Tensor, Dict]:
        x = self.fc(x)
        x = self.dropout(x)
        return F.log_softmax(x, dim=1), filters

class miniCNNEachNet(nn.Module):
    def __init__(self, params, num_class: int) -> None:
        '''
        Args:
            params (dict):
        Returns:
            None
        '''
        super(miniCNNEachNet, self).__init__()
        if params['encode'] == 'deep_hla':
            fc_len = params['fc_len']
        else:
            fc_len = params['emb_dim']
        dropout = params['dropout']
        mini_fc_len = params['mini_fc_len']
        if params['task']=='rnn':
            self.fc1 = nn.Linear(fc_len*2, mini_fc_len)
        elif params['encode'] == 'input_conv':
            self.fc1 = nn.Linear(params['emb_dim'] * params['chunk_split_num'], mini_fc_len)
        else:
            self.fc1 = nn.Linear(fc_len, mini_fc_len)
        self.bn1 = nn.BatchNorm1d(mini_fc_len)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(mini_fc_len, mini_fc_len)
        self.bn2 = nn.BatchNorm1d(mini_fc_len)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(mini_fc_len, num_class)
        self.convert_identity = nn.Linear(fc_len, mini_fc_len)

    def forward(self, x, filters) -> Tuple[Tensor, Dict]:
        #identity = x.clone()
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        #x = self.fc2(x)
        #x = self.bn2(x)
        #x = self.relu2(x)
        #x = self.dropout2(x)
        #identity = self.convert_identity(identity)
        #x += identity
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), filters

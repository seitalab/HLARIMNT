
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from argparse import Namespace
from torch import Tensor
from typing import Tuple, Dict

class BiRNNEachNet(nn.Module):
    def __init__(self, params, num_class: int) -> None:
        '''
        Args:
            params (dict):
        Returns:
            None
        '''
        super(BiRNNEachNet, self).__init__()
        fc_len = params['emb_dim'] * 2
        dropout = params['dropout']

        self.fc = nn.Linear(fc_len, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, filters) -> Tuple[Tensor, Dict]:
        x = self.fc(x)
        x = self.dropout(x)
        return F.log_softmax(x, dim=1), filters

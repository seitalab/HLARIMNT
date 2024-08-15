#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from torch import Tensor
from argparse import Namespace
from typing import Tuple, Dict

class CNNEachNet(nn.Module):
    def __init__(self, params: Dict, num_class: int) -> None:
        '''
        Args:
            args (Namespace):
            num_class (int) : # of classes
        Returns:
            None
        '''
        super(CNNEachNet, self).__init__()
        fc_len = params['fc_len']
        self.fc = nn.Linear(fc_len, num_class)

    def forward(self, x, filters) -> Tuple[Tensor, Dict]:
        if filters['mask_fc'] is None:
            filters['mask_fc'] = torch.bernoulli(x.data.new(x.data.size()).fill_(0.5))
        if self.training:
            x = x * filters['mask_fc']
        x = self.fc(x)
        assert filters['mask_fc'] is not None
        return F.log_softmax(x, dim=1), filters

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from torch import Tensor
from argparse import Namespace
from typing import Tuple, Dict

class CNNShared(nn.Module):
    def __init__(self, params, input_len) -> None:
        """
        Args:
            params:
        Returns:
            None:
        """
        super(CNNShared, self).__init__()

        conv1_num_filter = params["conv1_num_filter"]
        conv2_num_filter = params["conv2_num_filter"]
        conv1_kernel_size = params["conv1_kernel_size"]
        conv2_kernel_size = params["conv2_kernel_size"]
        fc_len = params["fc_len"]
        linear_input = (((input_len - conv1_kernel_size + 1) // 2) - conv2_kernel_size + 1) // 2

        self.input_collapse = params['input_collapse']
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2, stride=2)
        self.conv1 = nn.Conv1d(2, conv1_num_filter, kernel_size=conv1_kernel_size)
        self.conv2 = nn.Conv1d(conv1_num_filter, conv2_num_filter, kernel_size=conv2_kernel_size)
        self.bn1 = nn.BatchNorm1d(conv1_num_filter)
        self.bn2 = nn.BatchNorm1d(conv2_num_filter)
        self.fc = nn.Linear(conv2_num_filter * linear_input, fc_len)

    def forward(self, x, filters) -> Tuple[Tensor, Dict]:
        """
        Args:
            x:
            params:
        Returns:
            x, params
            """
        x = x.transpose(1, 2)
        if self.training and self.input_collapse:
            if filters['mask_input'] is None:
                filters['mask_input'] = torch.bernoulli(x.data.new(x.data.size()).fill_(random.uniform(0.8, 1)))
            x = x * filters['mask_input']

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        if filters['mask_conv1'] is None:
            filters['mask_conv1'] = torch.bernoulli(x.data.new(x.data.size()).fill_(0.5))

        if self.training:
            x = x * filters['mask_conv1']

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        if filters['mask_conv2'] is None:
            filters['mask_conv2'] = torch.bernoulli(x.data.new(x.data.size()).fill_(0.5))

        if self.training:
            x = x * filters['mask_conv2']

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        assert filters['mask_conv1'] is not None
        assert filters['mask_conv2'] is not None
        #assert filters['mask_input'] is not None
        return x, filters

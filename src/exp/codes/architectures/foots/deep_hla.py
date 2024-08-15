
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from argparse import Namespace
from torch import Tensor
from typing import Tuple, Dict

class CNNFoot(nn.Module):
    def __init__(self, settings, input_len, chunk_len) -> None:
        '''
        Args:
            params (Namespace) : 
        Returns:
            None
        '''
        super(CNNFoot, self).__init__()
        pass 
    
    def forward(self, x: Tensor, filters) -> Tuple[Tensor, Dict]:
        '''
        Args:
            x (Tensor) : Tensor of size (batchsize, num_ref, 2)
        Returns:
            x (Tensor) Tensor of size (batchsize, emb_dim, ): 
        '''
        return x, filters

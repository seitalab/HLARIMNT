
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from argparse import Namespace
from torch import Tensor
from typing import Tuple, Dict

class SinePE(nn.Module):
    def __init__(self, params, input_len) -> None:
        """
        Args:
            params (dict) :
        Returns:
            None
        """
        super(SinePE, self).__init__()

        pe_dropout = params['pe_dropout']
        emb_dim = params['emb_dim']
        if params['encode'] == 'input_conv':
            input_len = params['chunk_split_num']
        """
        den = torch.exp(- torch.arange(0, emb_dim, 2) * math.log(10000) / emb_dim)
        pos = torch.arange(0, input_len).reshape(input_len, 1)
        pos_embedding = torch.zeros((input_len, emb_dim))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        """
        pe = torch.zeros(input_len, emb_dim)
        #print('pe:',pe.shape)
        position = torch.arange(0, input_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))

        pe[:, 0::2] = torch.sin(position * div_term) * 1.0#params.pe_multiply
        pe[:, 1::2] = torch.cos(position * div_term) * 1.0#params.pe_multiply
        #pe = pe.unsqueeze(0).transpose(0, 1)
        pe = torch.reshape(pe,(1,pe.shape[0], pe.shape[1]))
        #print('pe after:',pe.shape)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(pe_dropout)
        #self.pos_embedding = pos_embedding
        #self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x, filters) -> Tuple[Tensor, Dict]:
        """
        Args:
            x (Tensor): Tensor of size (batchsize, emb_dim, )
        Returns:
            x (Tensor) : Tensor of size(batchsize, emb_dim, )
        """
        """
        self.pos_embedding = self.pos_embedding.to(x.device)
        x += self.pos_embedding[:x.size(0),:]
        x = self.dropout(x)
        return x, filters
        """
        #print('x:', x.shape)
        #print('tasuyatu:', self.pe[:x.size(0),:].shape)
        x = x + self.pe#[:x.size(0), :]
        return self.dropout(x), filters

class CNNPE(nn.Module):
    def __init__(self, settings, input_len) -> None:
        """
        Args:
            params (dict) :
        Returns:
            None
        """
        super(CNNPE, self).__init__()
        pass

    def forward(self, x, filters) -> Tuple[Tensor, Dict]:
        """
        Args:
            x (Tensor): Tensor of size (batchsize, emb_dim, )
        Returns:
            x (Tensor) : Tensor of size(batchsize, emb_dim, )
        """
        return x, filters

class LinearPE(nn.Module):

    def __init__(self, params, input_len) -> None:
        super(LinearPE, self).__init__()

        pe_dropout = params['pe_dropout']
        emb_dim = params['emb_dim']
        self.dropout = nn.Dropout(p=pe_dropout)

        stddev = 0.02
        # Tensor with (mean = 0, stddev = stddev)
        w = torch.randn(input_len+1, emb_dim) * (stddev ** 0.5)
        w = w.unsqueeze(0)
        self.pe = nn.Parameter(w)

    def forward(self, x: torch.Tensor, filters):
        """
        Add positional encoding.
        Args:
            x (torch.Tensor): Tensor of size (batch_size, num_steps, d_model).
        Returns:
            x (torch.Tensor): Tensor of size (batch_size, num_steps, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x), filters
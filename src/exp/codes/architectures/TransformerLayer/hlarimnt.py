
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from argparse import Namespace
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor

class TransformerLayer(nn.Module):
    def __init__(self, params, input_len) -> None:
        """
        Args:
            params (dict): Settings.
            input_len: Length of the input, which represents "chunk_num+1".
        Returns:
            None
        """
        super(TransformerLayer, self).__init__()

        emb_dim = params['emb_dim']
        dim_feedforward = params['dim_feedforward']
        nhead = params['nhead']
        dropout = params['dropout']
        num_layers = params['num_layers']

        self.pe = SinePE(params, input_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, dim_feedforward=dim_feedforward, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of size (batch_size, emb_dim,)
        Returns:
            x (Tensor): Tensor of size (batch_size, emb_dim,)
        """

        x = self.pe(x)
        x = self.encoder(x)
        x = x[:,0,:]
        return x

class SinePE(nn.Module):
    def __init__(self, params, input_len) -> None:
        """
        Args:
            params (dict) :
            input_len (int) :
        Returns:
            None
        """
        super(SinePE, self).__init__()

        pe_dropout = params['pe_dropout']
        emb_dim = params['emb_dim']

        pe = torch.zeros(input_len, emb_dim)
        position = torch.arange(0, input_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.reshape(pe,(1,pe.shape[0], pe.shape[1]))

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(pe_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of size (batchsize, emb_dim, )
        Returns:
            x (Tensor) : Tensor of size (batchsize, emb_dim, )
        """

        x = x + self.pe
        return self.dropout(x)

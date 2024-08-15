
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from argparse import Namespace
from torch import Tensor
from typing import Tuple, Dict
import pandas as pd
import yaml

class PrimeEmbed(nn.Module):
    def __init__(self, params, input_len, chunk_len) -> None:
        '''
        Args:
            params (Namespace) : 
        Returns:
            None
        '''
        super(PrimeEmbed, self).__init__()
        self.params = params
        emb_dim = params['emb_dim']
        encode_type = params['encode']
        kmeans = params['kmeans']

        if encode_type == 'by_input_len':
            self.embedding = nn.Embedding(input_len * 2 + 2, emb_dim)
        elif encode_type == 'by_input_len_base':
            self.embedding = nn.Embedding(input_len * 8 + 2, emb_dim)
        elif encode_type == '2dim':
            self.embedding = nn.Embedding(4, emb_dim)
        elif encode_type == '4dim':
            self.embedding = nn.Embedding(6, emb_dim)
        elif encode_type == '8dim':
            self.embedding = nn.Embedding(10, emb_dim) 
        elif encode_type == 'input_conv':
            chunk_spilt_num = params['chunk_split_num']
            stride = 50
            print('input_len:',input_len)
            print(input_len-(chunk_spilt_num-1)*stride)
            self.embedding = nn.Conv1d(2, params['emb_dim'], kernel_size=input_len-(chunk_spilt_num-1)*stride, stride=stride)
        elif encode_type == 'chunk':
            config_file = '../config.yaml'

            self.embedding_conv =nn.Conv1d(2, params['embed_conv_dim'], kernel_size=chunk_len)
            self.bn1 = nn.BatchNorm1d(params['embed_conv_dim'])
            self.relu = nn.ReLU()
            self.embedding_fc = nn.Linear(params['embed_conv_dim']*chunk_len, emb_dim)
            self.embedding = nn.Linear(2*chunk_len, emb_dim)
            #self.cls_emb = nn.Linear(2*chunk_len, emb_dim)

    def forward(self, x: Tensor, filters) -> Tuple[Tensor, Dict]:
        '''
        Args:
            x (Tensor) : Tensor of size (batchsize, num_ref, 2)
        Returns:
            x (Tensor) Tensor of size (batchsize, emb_dim, ): 
        '''

        if self.params['encode'] == 'chunk' and not self.params['input_conv']:
            #bs * input_len * chunk_len * 2 -> bs * input_len * (2.chunk_len)
            x = x.view(x.size()[0], x.size()[1], -1)
            #print('after view:', x.shape)
            #if self.params['cls_token_sep']:
                #x = torch.cat((self.embedding(x[:,1:,:]), self.cls_emb(x[:,0,:].unsqueeze_(1))), dim=1)
            #else:
            x = self.embedding(x)
            #print('final:',x.shape)
        elif self.params['encode'] == 'chunk' and self.params['input_conv']:
            #bs * input_len * chunk_len * 2 -> bs * input_len * (2.chunk_len)
            for i in range(x.shape[1]):
                x[:,i,:,:] = self.embedding_conv(x[:,i,:,:])
            #x = self.embedding_conv(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = x.view(x.size()[0], x.size()[1], -1)
            x = self.embedding_fc(x)

        elif self.params['encode'] == 'input_conv':
            x = x.transpose(1, 2)
            #print(x.shape)
            x = self.embedding(x)
            x = x.transpose(1, 2)

        else:
            x = x.to(torch.int64)
            x = self.embedding(x)
        return x, filters

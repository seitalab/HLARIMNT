
import torch.nn as nn
from torch import Tensor

class EmbeddingLayer(nn.Module):
    def __init__(self, params: dict, input_len: int, chunk_len: int) -> None:
        """
        Args:
            params (dict): Settings.
            input_len (int): Length of the input.
            chunk_len: (int): Numbers of SNPs one chunk has.
        Returns:
            None
        """
        super(EmbeddingLayer, self).__init__()
        emb_dim = params['emb_dim']
        self.embedding = nn.Linear(2*chunk_len, emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor) : Tensor of size (batch_size, chunk_num+1, chunk_len, 2)
        Returns:
            x (Tensor) Tensor of size (batch_size, emb_dim, ): 
        """
        x = x.view(x.size()[0], x.size()[1], -1)
        x = self.embedding(x)
        return x

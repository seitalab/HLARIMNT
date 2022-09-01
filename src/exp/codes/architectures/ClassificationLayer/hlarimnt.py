
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ClassificationLayer(nn.Module):
    def __init__(self, params, class_num: int) -> None:
        """
        Args:
            params (dict): Settings.
            class_num (int): Numbers of alleles the gene has
        Returns:
            None
        """
        super(ClassificationLayer, self).__init__()
        emb_dim = params['emb_dim']
        dropout = params['dropout']
        self.fc = nn.Linear(emb_dim, class_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of size (batch_size, emb_dim)
        Returns:
            x (Tensor): Tensor of size (batch_size, class_num)
        """
        x = self.fc(x)
        x = self.dropout(x)
        return F.log_softmax(x, dim=1)

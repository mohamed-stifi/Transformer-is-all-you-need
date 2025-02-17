import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        # create matrix of shape (max_len, embedding_dim)
        pe = torch.zeros(self.max_len, self.embedding_dim)

        # create vector of shape (max_len, 1)
        pos = torch.arange(0, max_len, dtype= torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float()*(-math.log(10000.0)/self.embedding_dim))

        pe[:, 0::2] = torch.sin(pos*div_term)
        pe[:, 1::2] = torch.cos(pos*div_term)

        pe = pe.unsqueeze(0)   # (1, max_len, embedding_dim)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
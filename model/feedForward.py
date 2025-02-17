import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout):
        super(FeedForward, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        # (batch_s, seq_len, embedding_dim) --> (batch_s, seq_len, hidden_dim) --> (batch_s, seq_len, embedding_dim)
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

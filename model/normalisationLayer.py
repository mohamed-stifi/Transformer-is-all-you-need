import torch.nn as nn
import torch

class LayerNormalisation(nn.Module):
    def __init__(self, eps = 10**-6):
        super(LayerNormalisation, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bais = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bais
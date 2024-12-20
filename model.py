from torch import Tensor, nn
import torch
from torch.nn import functional as F

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k, model):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])

        

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            elif len(tokens[i].shape) == 2:
                tokens[i] = self.fc[i](tokens[i])
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens
    







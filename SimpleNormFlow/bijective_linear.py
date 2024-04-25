from .bijection import _Bijection
import torch.nn as nn
import torch


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


class BijectiveLinear(_Bijection):
    def __init__(self, dim):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(_Bijection, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim, device=device)) # DxD
        self.bias = nn.Parameter(torch.zeros(1, dim, device=device)) # 1xD

    def forward(self, x):   # x has shape NxD
        z = x @ self.weight.t() + self.bias
        log_abs_det = sum_except_batch(torch.slogdet(self.weight)[1])
        return z, log_abs_det # shapes NxD, N

    def inverse(self, z):
        # ...
        x = (z - self.bias) @ torch.linalg.inv(self.weight.T)
        return x    # NxD

    def regularization(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ((self.weight @ self.weight) - torch.eye(self.dim, device=device)).abs().sum()



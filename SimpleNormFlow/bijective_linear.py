from SimpleNormFlow.bijection import _Bijection
import torch.nn as nn
import torch


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


class BijectiveLinear(_Bijection):
    def __init__(self, dim):
        super(_Bijection, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim)) # DxD
        self.bias = nn.Parameter(torch.zeros(1, dim)) # 1xD

    def forward(self, x):   # x has shape NxD
        # ...
        N = x.shape[0]

        z = x @ self.weight.T + self.bias
        log_abs_det = torch.log(torch.abs(torch.linalg.det(self.weight))).repeat(N)
        return z, log_abs_det # shapes NxD, N

    def inverse(self, z):
        # ...
        x = (z - self.bias) @ torch.linalg.inv(self.weight.T)
        return x    # NxD

    def regularization(self):
        return ((self.weight @ self.weight) - torch.eye(self.dim)).abs().sum()



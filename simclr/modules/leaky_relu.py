import torch.nn as nn


class LeakyReLU(nn.Module):

    def __init__(self):
        super(LeakyReLU, self).__init__()

    def forward(self, x):
        return nn.functional.leaky_relu(x)

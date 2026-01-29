import torch
import torch.nn as nn
from torch.nn import Parameter


class JaxGRUCell(nn.Module):
    """
    This GRU is a torch version of jax gru in fixed_point_finder.
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.bC = Parameter(torch.Tensor(hidden_size))
        self.bRU = Parameter(torch.Tensor(2 * hidden_size))

        self.wCX = Parameter(torch.Tensor(hidden_size, input_size))
        self.wCH = Parameter(torch.Tensor(hidden_size, hidden_size))

        self.h0 = Parameter(torch.Tensor(hidden_size))

        self.wRUX = Parameter(torch.Tensor(2 * hidden_size, input_size))
        self.wRUH = Parameter(torch.Tensor(2 * hidden_size, hidden_size))

        self.init_weights()

    def init_weights(self):
        h_factor = 1.0 / self.hidden_size
        # i_factor = 1.0 / self.input_size

        self.bC.data.zero_()

        self.wCX.data.uniform_(-h_factor, h_factor)
        self.wCH.data.uniform_(-h_factor, h_factor)

        self.wRUX.data.uniform_(-h_factor, h_factor)
        self.wRUH.data.uniform_(-h_factor, h_factor)

        self.bRU.data.uniform_(-h_factor, h_factor)
        self.h0.data.uniform_(-h_factor, h_factor)

    def forward(self, x, h, bfg=0.5):
        ## b,x
        hx = torch.cat([h, x], dim=1)
        wRUHX = torch.cat([self.wRUX, self.wRUH], dim=1)
        ru = torch.sigmoid(hx @ wRUHX.T + self.bRU[None, ...])

        r, u = torch.split(ru, int(ru.shape[1] / 2), dim=1)
        rhx = torch.cat([r * h, x], dim=1)

        wCHX = torch.cat([self.wCX, self.wCH], dim=1)
        c = torch.tanh(rhx @ wCHX.T + self.bC[None, ...] + bfg)
        return u * h + (1 - u) * c

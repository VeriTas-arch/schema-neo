import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init


class RNN(nn.RNN):
    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__(*args, **kwargs)


class LeakyRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        tau_trainable=False,
        tau=10,
        act_fn="tanh",
        bias=True,
    ):
        super(LeakyRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if act_fn == "tanh":
            self.act_fn = F.tanh
        elif act_fn == "relu":
            self.act_fn = F.relu

        self.dt = 1
        self.tau_trainable = tau_trainable

        self.win = Parameter(torch.Tensor(input_size, hidden_size))
        self.wr = Parameter(torch.Tensor(hidden_size, hidden_size))
        if tau_trainable:
            self.tau = Parameter(torch.Tensor(1, hidden_size))
        else:
            self.tau = tau

        if bias:
            self.bias = Parameter(torch.Tensor(1, hidden_size))
        else:
            self.bias = 0

        self.init_weights()

    def init_weights(self):
        ### for input weights
        init.kaiming_normal_(self.win, a=np.sqrt(5))

        ### for recurrent weights
        stdv = 1.0 / np.sqrt(self.hidden_size) * 0.1
        init.uniform_(self.wr, -stdv, stdv)

        if self.tau_trainable:
            init.uniform_(self.tau, 4, 20)

        if isinstance(self.bias, torch.Tensor):
            init.zeros_(self.bias)

    def one_step(self, x, h):
        ## x, input tensor, shape (batch,n_input)
        ### dh = -h + f(win*x + b + wrec*h)

        dh = (
            -h
            + self.act_fn(
                torch.matmul(h, self.wr) + torch.matmul(x, self.win) + self.bias
            )
        ) / self.tau

        h = h + self.dt * dh

        return h

    def forward(self, x, h=None):
        seq, batch, f_dim = x.shape

        if h is None:
            h = torch.zeros(batch, self.hidden_size, dtype=x.dtype, device=x.device)

        outs = torch.zeros((seq, batch, self.hidden_size)).to(x.device)
        for t in range(seq):
            h = self.one_step(x[t], h)
            outs[t, :, :] = h

        return outs

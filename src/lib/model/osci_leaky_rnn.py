import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


class OsciLeakyRNN(nn.modules.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        osci_fn=None,
        add_oscillation=False,
        act_fn="tanh",
        tau_learnable=False,
        bias=True,
        osci_scale=1.0,
        osci_omeg=20,
        **kwargs
    ):

        super(OsciLeakyRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_learnable = tau_learnable
        self.add_oscillation = add_oscillation

        self.osci_scale = osci_scale
        self.osci_omeg = osci_omeg

        if act_fn == "tanh":
            self.act_fn = F.tanh
        elif act_fn == "relu":
            self.act_fn = F.relu

        self.win = Parameter(torch.Tensor(input_size, hidden_size))
        self.wr = Parameter(torch.Tensor(hidden_size, hidden_size))

        if bias:
            self.bias = Parameter(torch.Tensor(1, hidden_size))
        else:
            self.bias = 0

        if tau_learnable:
            self.alpha = Parameter(torch.Tensor(1, hidden_size))
        else:
            self.dt = kwargs.get("dt")
            self.tau = kwargs.get("tau")
            self.alpha = self.dt / self.tau

        self.init_weights()

    def init_weights(self):
        ### for input weights
        init.kaiming_normal_(self.win, a=np.sqrt(5))

        ### for recurrent weights
        stdv = 1.0 / np.sqrt(self.hidden_size)
        init.uniform_(self.wr, -stdv, stdv)

        init.zeros_(self.bias)

        ###
        if self.tau_learnable:
            init.uniform_(self.alpha, 10 / 1000, 10 / 100)

    @property
    def osci_curr(self):
        t = np.arange(500)
        osci_curr = torch.sin(torch.FloatTensor(t * 2 * np.pi / self.osci_omeg))
        return osci_curr * self.osci_scale

    def one_step(self, x, osci_x, hidden):
        ## x, input tensor, shape (batch,n_input)
        ## hidden, hidden states, a list

        ## tau*dr/dt = -r + f(wr*r+win*xin+bias)
        ## r_new = (1-alpha)*r_old + alpha*f(wr*r_old+win*xin+bias)

        (hidden_old,) = hidden
        tot_current = (
            osci_x
            + torch.matmul(hidden_old, self.wr)
            + torch.matmul(x, self.win)
            + self.bias.to(x.device)
        )
        hidden_new = (
            self.act_fn(tot_current) * self.alpha + (1 - self.alpha) * hidden_old
        )
        return hidden_new, (hidden_new,)

    def forward(self, x, hx=None):
        seq, batch, f_dim = x.shape
        if hx is None:
            zeros = torch.zeros(batch, self.hidden_size, dtype=x.dtype, device=x.device)
            hx = (zeros,)

        outs = []
        for t in range(seq):
            if self.add_oscillation:
                osci_xt = self.osci_curr[t].to(x.device)
            else:
                osci_xt = 0
            out_t, hx = self.one_step(x[t], osci_xt, hx)
            outs.append(out_t)

        outs = torch.stack(outs, dim=0)
        return outs, hx

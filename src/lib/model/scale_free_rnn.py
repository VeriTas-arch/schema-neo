import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


def get_scale_free_mat(p1, N):
    Cmat = np.zeros((N, N))  ## symmetry connection matrix
    for i in range(N):
        if np.random.rand() > p1:
            nc = 2
        else:
            nc = 1

        if i == 0:
            continue  ## fist node, no connections
        if i == 1:
            Cmat[0, 1] = Cmat[1, 0] = 1
            continue
        nc_vec = np.sum(Cmat[:i, :i], axis=1)
        nc_vec_norm = nc_vec / np.sum(nc_vec)
        nc_vec_cusum = np.cumsum(nc_vec_norm)
        for h in range(nc):
            j = np.sum(np.random.rand() > nc_vec_cusum)
            Cmat[i, j] = 1
            Cmat[j, i] = 1
    return Cmat


class ScaleFreeRNN(nn.modules.Module):
    """
    Build the scale-free structure recurrent neural network.
    simiar to the graph neural network.

    Args:
    1. mode: "raw", means raw_rnn, simple nonlinear, without single neural dynamics
             "rate", means rate_rnn, with single neural dynamics
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        act_fn="tanh",
        mode="raw",
        mask_mode="all2all",
        tau_learnable=False,
        bias=True,
        subgroup=1,
        **kwargs
    ):
        super(ScaleFreeRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.win = Parameter(torch.Tensor(hidden_size, input_size))
        self.wr = Parameter(torch.Tensor(hidden_size, hidden_size))

        if bias:
            self.bias = Parameter(torch.Tensor(1, hidden_size))
        else:
            self.bias = 0

        self.subgroup = subgroup
        self.n_groups = int(hidden_size / subgroup)

        self.mode = mode
        if act_fn == "tanh":
            self.act_fn = F.tanh
        elif act_fn == "relu":
            self.act_fn = F.relu
        else:
            raise ValueError("activation function is not defined, ", act_fn)
        if mode == "rate":
            self.alpha = kwargs.get("alpha")  ## dt/tau

        self.init_weights()
        self.mask_matrix = self.set_topology(topology_mode=mask_mode, **kwargs)
        self.mask_matrix.requires_grad = False

    def set_topology(self, topology_mode="all2all", **kwargs):
        """
        build the scale free neural network.
        return the connection matrix.
        """
        if topology_mode == "uniform_random":
            p = kwargs.get("p")
            weight_mask = torch.rand(self.n_groups, self.n_groups) > p
            weight_mask = (
                weight_mask.view(-1, 1)
                .repeat(1, self.subgroup)
                .view(self.n_groups, 1, self.hidden_size)
                .repeat(1, self.subgroup, 1)
                .view(self.hidden_size, self.hidden_size)
            )
        elif topology_mode == "scale_free":
            p1 = kwargs.get("p")
            weight_mask = get_scale_free_mat(p1, self.n_groups)
            weight_mask = torch.FloatTensor(weight_mask)
            weight_mask = (
                weight_mask.view(-1, 1)
                .repeat(1, self.subgroup)
                .view(self.n_groups, 1, self.hidden_size)
                .repeat(1, self.subgroup, 1)
                .view(self.hidden_size, self.hidden_size)
            )
        elif topology_mode == "all2all":
            weight_mask = np.ones((self.n_groups, self.n_groups)) - np.identity(
                self.n_groups
            )
            weight_mask = torch.FloatTensor(weight_mask)
            weight_mask = (
                weight_mask.view(-1, 1)
                .repeat(1, self.subgroup)
                .view(self.n_groups, 1, self.hidden_size)
                .repeat(1, self.subgroup, 1)
                .view(self.hidden_size, self.hidden_size)
            )
        elif topology_mode == "small_world":
            raise NotImplementedError("This topology has not been implemented!")
        else:
            raise ValueError("This topology structure is not defined !", topology_mode)
        return weight_mask

    def init_weights(self):
        ### for input weights
        init.kaiming_uniform_(self.win, a=np.sqrt(5))

        ### for recurrent weights
        stdv = 1.0 / np.sqrt(self.hidden_size)
        init.uniform_(self.wr, -stdv, stdv)

        init.zeros_(self.bias)

    def one_step(self, x, hidden):
        ### hidden is a tuple,
        self.mask_matrix = self.mask_matrix.to(x.device)
        if self.mode == "raw":
            (hidden,) = hidden
            out = self.act_fn(
                torch.matmul(hidden, self.wr.t() * self.mask_matrix)
                + torch.matmul(x, self.win.t())
                + self.bias.to(x.device)
            )
            return out, (out,)

        if self.mode == "rate":
            ### ri(t+1) = (1-alpha)*ri(t) + (wr*act[ri(t)] + win*x(t))*alpha
            ### hidden , here is recurrent output
            r_pre, r_post = hidden
            r_pre = (1 - self.alpha) * r_pre + (
                torch.matmul(r_post, self.wr.t() * self.mask_matrix)
                + torch.matmul(x, self.win.t())
                + self.bias.to(x.device)
            ) * self.alpha
            r_post = self.act_fn(r_pre)
            return r_post, (r_pre, r_post)

    def forward(self, x, hx=None):
        ###
        seq, batch, f_dim = x.shape

        if hx is None:
            zeros = torch.zeros(batch, self.hidden_size, dtype=x.dtype, device=x.device)
            if self.mode == "rate":
                hx = (zeros, zeros)
            if self.mode == "raw":
                hx = (zeros,)
        outs = []
        for t in range(seq):
            out_t, hx = self.one_step(x[t], hx)
            outs.append(out_t)
        outs = torch.stack(outs, dim=0)
        return outs, hx


###

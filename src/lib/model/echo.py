import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SimpleEcho(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=10,
        tau=10,
        dt=1,
        rho=1.3,
        spars_echo=0.1,
        spars_inp=0.1,
        tanh_slop=1.0,
        scale_inp=1.0,
        with_rnn_dynamics=True,
        activation=F.tanh,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = dt / tau
        self.with_rnn_dynamics = with_rnn_dynamics
        self.rho = rho
        self.act = activation
        self.spars_echo = spars_echo
        self.spars_inp = spars_inp
        self.scale_inp = scale_inp
        self.tanh_slop = tanh_slop
        print("tanh_slop is ", tanh_slop)

        self.win = Parameter(torch.Tensor(out_channels, in_channels))
        self.wr = Parameter(torch.Tensor(out_channels, out_channels))

        # pdb.set_trace()

        self.init_weights()

    def init_weights(self):
        """
            wr, reservoir network structure; not trainable.
            win, uniform distribution; not trainable.


            J = np.random.randn(N, N).astype(np.float32)
        p_mar = np.random.rand(N, N).astype(np.float32)
        p_mar = p_mar < p_con
        J = J * p_mar
        numx = 1 - dt/tau
        M = dt/tau * J + numx * np.eye(N).astype(np.float32)
        rho_all, _ = np.linalg.eig(M)
        rho = max(np.abs(rho_all))

        J = J / (rho - numx) * (rho_scale - numx)
        """
        # wr = np.random.rand(self.out_channels,self.out_channels) - 0.5
        wr = np.random.randn(self.out_channels, self.out_channels)
        wr[np.random.rand(*wr.shape) > self.spars_echo] = 0.0
        M = np.eye(self.out_channels) * (1.0 - self.alpha) + wr * self.alpha
        radius = np.max(np.abs(np.linalg.eigvals(M)))
        wr = wr / (radius - 1.0 + self.alpha) * (self.rho - 1.0 + self.alpha)
        self.wr.data = torch.FloatTensor(wr)
        self.wr.requires_grad = False

        win = np.random.uniform(
            low=-self.scale_inp,
            high=self.scale_inp,
            size=(self.out_channels, self.in_channels),
        )
        win[np.random.rand(*win.shape) > self.spars_inp] = 0.0

        self.win.data = torch.FloatTensor(win)
        self.win.requires_grad = False

    def forward(self, x, hid):
        u, h = hid
        if self.with_rnn_dynamics:
            self.alpha = torch.FloatTensor([self.alpha]).to(x.device)
            u_new = u + self.alpha * (-u + F.linear(x, self.win) + F.linear(h, self.wr))
        else:
            u_new = u + self.alpha * (-u + F.linear(x, self.win))
            # u_new = F.linear(x,self.win)
        h_new = self.act(u_new * self.tanh_slop)
        return h_new, (u_new, h_new)

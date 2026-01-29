import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .utils import LogAct, RecLogAct, bi_relu


class TrainDMCell(nn.Module):

    def __init__(
        self,
        in_channels=5,
        out_channels=2,
        Je=8.0,
        Jm=-2,
        I0=0.0,
        dt=1.0,
        taus=100.0,
        gamma=0.1,
        wr_trainable=True,
        taus_trainable=True,
        I0_trainable=True,
        activation=LogAct(),
        rec_activation=RecLogAct(),
    ):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.Je = Je
        self.Jm = Jm
        self.I0 = I0
        self.dt = dt
        self.taus_value = taus
        self.gamma = gamma

        self.wr_trainable = wr_trainable
        self.I0_trainable = I0_trainable
        self.taus_trainable = taus_trainable

        self.act = activation
        self.rec_act = rec_activation

        ## tau and wr need mask
        self.win = Parameter(torch.Tensor(out_channels, in_channels))
        self.I0 = Parameter(torch.Tensor(out_channels))
        self.taus = Parameter(torch.Tensor(out_channels))
        self.wr = Parameter(torch.Tensor(out_channels, out_channels))
        self.mask = torch.FloatTensor(
            -np.ones((out_channels, out_channels)) + np.eye(self.out_channels) * 2
        )

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / np.sqrt(self.out_channels)
        nn.init.uniform_(self.win, -stdv, stdv)

        wr = np.ones((self.out_channels, self.out_channels)) * self.Jm
        wr = (
            wr
            + np.eye(self.out_channels) * self.Je
            - np.eye(self.out_channels) * self.Jm
        )
        self.wr.data = torch.FloatTensor(wr)
        self.I0.data = torch.FloatTensor(np.zeros(self.out_channels))
        self.taus.data = torch.FloatTensor([self.taus_value])

        if not self.wr_trainable:
            self.wr.requires_grad = False
        if not self.I0_trainable:
            self.I0.requires_grad = False
        if not self.taus_trainable:
            self.taus.requires_grad = False

    def forward(self, x, hid):
        """
        learning rule is "bp"
        """
        s = hid[0]
        alpha = self.dt / bi_relu(self.taus)
        r_input = F.linear(x, self.win)
        self.mask = self.mask.to(self.wr.device)
        wr = bi_relu(self.wr) * self.mask
        rec_input = self.I0 + F.linear(s, wr)
        rx = r_input + rec_input
        r = self.act(rx)
        s_new = s + alpha * (-s + (1.0 - s) * self.gamma * r)

        return r_input, rec_input, rx, r, (s_new,)


if __name__ == "__main__":
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .utils import LogAct, RecLogAct


class ForceDMCell(nn.Module):

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
        target_mode="feedforward_input",  ## total_input, output, feedforward_input
        force_interval=1,
        activation=LogAct(),
        rec_activation=RecLogAct(),
    ):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.Je = Je
        self.Jm = Jm
        self.I0 = I0
        self.alpha = dt / taus
        self.gamma = gamma
        self.win = Parameter(torch.Tensor(out_channels, in_channels))
        self.wr = Parameter(torch.Tensor(out_channels, out_channels))
        self.act = activation
        self.rec_act = rec_activation
        self.target_mode = target_mode
        self.force_interval = force_interval
        self.softmax = nn.Softmax()

        self.init_weights()

    def init_weights(self):
        # self.win.data = torch.zeros((self.out_channels,self.in_channels))
        self.win.data = torch.eye(self.out_channels, self.in_channels)
        self.win.requires_grad = False

        wr = np.ones((self.out_channels, self.out_channels)) * self.Jm
        wr = (
            wr
            + np.eye(self.out_channels) * self.Je
            - np.eye(self.out_channels) * self.Jm
        )
        self.wr.data = torch.FloatTensor(wr)
        self.wr.requires_grad = False

    def forward(self, x, hid, y=None, update_steps=0):
        """
        learning rule is "force" or "bp"
        """
        self.alpha = torch.FloatTensor([self.alpha]).to(x.device)

        # pdb.set_trace()
        if not self.training:
            s = hid[0]
            r_input = F.linear(x, self.win)
            rec_input = self.I0 + F.linear(s, self.wr)
            rx = r_input + rec_input
            r = self.act(rx)

            s_new = s + self.alpha * (-s + (1.0 - s) * self.gamma * r)

            if y is None:
                y = 0
            if self.target_mode == "feedforward_input":
                err = r_input - y

            # return err, r_input,rec_input,rx,r,self_inh,self_exc,(s_new,)
            return err, r_input, rec_input, rx, r, (s_new,)

        else:
            batch_size = x.shape[0]
            s, P = hid  ## s(n-1); P(n-1)

            r_input = F.linear(x, self.win)

            rec_input = self.I0 + F.linear(s, self.wr)
            rx = r_input + rec_input

            # if self.target_mode == "total_input":
            # 	err = rx - y
            if self.target_mode == "feedforward_input":
                err = r_input - y
            ### compute the feedforward_input target and compute error
            # if self.target_mode == "neural_output":
            # 	y = self.rec_act(y)
            # 	err = rx - y

            if update_steps % self.force_interval == 0:
                r = x
                k_fenmu = F.linear(r, P)
                rPr = torch.sum(k_fenmu * r, 1, True)

                # print(P.mean())

                k_fenzi = 1.0 / (1.0 + rPr)
                k = k_fenmu * k_fenzi

                kall = k[:, :, None].repeat(1, 1, self.out_channels)
                # kall = torch.repeat(k[:, :, None], (1, 1, self.out_channels))
                dw = -kall * err[:, None, :]
                self.win.copy_(self.win + torch.mean(dw, 0).transpose(1, 0))

                P = P - F.linear(k.t(), k_fenmu.t()) / batch_size

            r = self.act(rx)
            s_new = s + self.alpha * (-s + (1.0 - s) * self.gamma * r)

            return err, r_input, rec_input, rx, r, (s_new, P)


if __name__ == "__main__":
    pass
